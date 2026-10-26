"""
Summarize and plot a set of *frozen* training runs against a set of *full* training runs.

Each run is a result directory produced by ``scripts/parallel_frozen_training.py`` or
``scripts/parallel_full_training.py`` and contains the memory-mapped ``.dat`` metric files plus
the sidecar ``*.pkl`` metadata. Every run carries a ``(chip, seed)`` identity (parsed from the
leaf directory name ``{weight_seed}_{chip}`` or, failing that, from ``params.pkl``).

The tool focuses on two metrics -- **target error** and **classification accuracy** -- and

  * computes mean / std / min / max across runs, at *test* time and at the *best validation*,
    separately for frozen and full training,
  * computes the mean / std of the frozen->full improvement over runs that share the same
    ``(chip, seed)``,
  * prints a formatted statistics table to the console,
  * draws one summary plot per metric with four mean +/- std lines
    (frozen-train, frozen-validation, full-train, full-validation), with an optional rolling
    window to smooth the training curves.

Run as a script (``python scripts/summarize_runs.py``) to print the table and save PDF+PNG plots,
or import the functions from a notebook to render the plots inline -- see ``summarize_runs.ipynb``.
"""

import os
import re
import glob
import pickle
import warnings
import json


import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIG -- edit these, or import the functions and pass your own values.
# ---------------------------------------------------------------------------
# List the result directories (one per run) for each training regime.
FROZEN_PATHS = [
    '../dynapse_results/yinyang/frozen/201285_0',
    '../dynapse_results/yinyang/frozen/201285_1',
    '../dynapse_results/yinyang/frozen/201285_2',
    '../dynapse_results/yinyang/frozen/201286_0',
    '../dynapse_results/yinyang/frozen/201286_1',
    '../dynapse_results/yinyang/frozen/201286_2',
    # '../results/parallel-frozen-feedback-control-dynapse/.../<timestamp>/<seed>_<chip>/',
]
FULL_PATHS = [
    '../dynapse_results/yinyang/full/201285_0',
    '../dynapse_results/yinyang/full/201285_1',
    '../dynapse_results/yinyang/full/201285_2',
    '../dynapse_results/yinyang/full/201286_0',
    '../dynapse_results/yinyang/full/201286_1',
    '../dynapse_results/yinyang/full/201286_2',
    '../dynapse_results/yinyang/full/201287_0',
    '../dynapse_results/yinyang/full/201287_1',
    '../dynapse_results/yinyang/full/201287_2',
    '../dynapse_results/yinyang/full/201288_0',
    '../dynapse_results/yinyang/full/201288_1',
    '../dynapse_results/yinyang/full/201288_2',
    # '../results/parallel-feedback-control-dynapse/.../<timestamp>/<seed>_<chip>/',
]

# Rolling-window length applied to the *training* curves only (validation curves are coarse and
# left raw). 1 means no smoothing.
ROLLING_WINDOW = 100

# Validation row 0 is an artificial pre-training baseline (see training.py) -- drop it.
DROP_VAL_BASELINE = True

# Where to write the PDF/PNG plots when run as a script.
OUTPUT_DIR = './summary_plots'

# Optional explicit ``(chip, seed)`` override, keyed by result path, when neither the directory
# name nor params.pkl can be trusted, e.g. {'/path/to/run': (chip, seed)}.
RUN_LABELS = {}


# Two metrics we care about. 'better' encodes the direction of "best" and "improvement".
METRICS = {
    # 'ylim' applies to the training-curve line plots; 'bar_ylim' to the grouped bar charts
    # (zoomed past 0 so the small differences between regimes/chips/seeds are visible).
    'target_error':            {'label': 'Target error [Hz]',      'better': 'lower',  'yscale': 'log',    'ylim': (3, 10),     'bar_ylim': (3, 7)},
    'classification_accuracy': {'label': 'Classification accuracy', 'better': 'higher', 'yscale': 'linear', 'ylim': (0.05, 0.80), 'bar_ylim': (0.4, 0.8)},
}

# Colour palette reused from src/yin_yang_utils.py: blue=train, purple=validation.
# Full training uses the dark shades; frozen training uses lighter shades of the same hues.
COLOR_TRAIN = '#05538D'        # full train  (dark blue)
COLOR_VAL = '#5E3493'          # full val    (dark purple)
COLOR_TRAIN_LIGHT = '#7FB3DA'  # frozen train (light blue)
COLOR_VAL_LIGHT = '#AD93CC'    # frozen val   (light purple)


# ---------------------------------------------------------------------------
# Small helpers copied verbatim from src/yin_yang_utils.py so this analysis
# script stays importable without the hardware (samna) dependency chain.
# ---------------------------------------------------------------------------
def latexify(width_in_points=5.91, width_fraction=1.0, height_ratio=None, base_font_size=10):
    """Set up matplotlib's RC params to match the LaTeX document appearance used elsewhere."""
    fig_width = width_in_points * width_fraction
    if height_ratio is None:
        fig_height = fig_width
    else:
        fig_height = fig_width * height_ratio
    plt.rcParams.update({
        "figure.figsize": (fig_width, fig_height),
        "font.size": base_font_size,
        "axes.labelsize": base_font_size,
        'axes.titlesize': 'medium',
        "legend.fontsize": base_font_size - 2,
        "legend.title_fontsize": base_font_size - 2,
        "xtick.labelsize": base_font_size - 2,
        "ytick.labelsize": base_font_size - 2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def calculate_rolling_mean(data: np.ndarray, rolling_window: int = 1):
    """Backward-looking rolling mean and std of a 1D array (matches src/yin_yang_utils.py)."""
    data = np.asarray(data, dtype=float)
    rolling_mean = np.zeros_like(data)
    rolling_std = np.zeros_like(data)
    for i in range(len(data)):
        i1 = max(0, i - rolling_window)
        rolling_mean[i] = data[i1:i + 1].mean()
        rolling_std[i] = data[i1:i + 1].std()
    return rolling_mean, rolling_std


# ---------------------------------------------------------------------------
# Loading a single run.
# ---------------------------------------------------------------------------
_LEAF_RE = re.compile(r'^(\d+)_(\d+)$')

# split key -> (metadata pkl filename, .dat file prefix)
_SPLITS = {
    'train': ('trainlog.pkl', ''),
    'val':   ('validationlog.pkl', 'val_'),
    'test':  ('testlog.pkl', 'test_'),
}


def _load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _identify_run(path):
    """Return ``(chip, seed)`` for a run directory.

    Priority: explicit RUN_LABELS override -> leaf directory name ``{seed}_{chip}`` ->
    params.pkl (``chip_id_hidden`` / ``weight_seed``).
    """
    norm = os.path.normpath(path)
    if norm in RUN_LABELS:
        chip, seed = RUN_LABELS[norm]
        return int(chip), int(seed)
    if path in RUN_LABELS:
        chip, seed = RUN_LABELS[path]
        return int(chip), int(seed)

    leaf = os.path.basename(norm)
    m = _LEAF_RE.match(leaf)
    if m:
        seed, chip = int(m.group(1)), int(m.group(2))
        return chip, seed

    par = _load_pickle(os.path.join(path, 'params.pkl'))
    if par is not None:
        chip = getattr(par, 'chip_id_hidden', getattr(par, 'chip_id_out', None))
        seed = getattr(par, 'weight_seed', None)
        if chip is not None and seed is not None:
            return int(chip), int(seed)

    raise ValueError(
        f"Could not determine (chip, seed) for run '{path}'. Expected a leaf directory named "
        f"'<seed>_<chip>', a params.pkl, or an entry in RUN_LABELS."
    )


def _memmap_metric(path, prefix, name):
    """Memmap a metric ``.dat`` file, reading its shape from the split's sidecar pkl.

    Falls back to flat reshaping if the metadata is missing.
    """
    dat_path = os.path.join(path, f"{prefix}{name}.dat")
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Missing metric file: {dat_path}")

    shape = None
    # Find the sidecar pkl matching this prefix and read the recorded shape.
    for pkl_name, pkl_prefix in _SPLITS.values():
        if pkl_prefix != prefix:
            continue
        meta = _load_pickle(os.path.join(path, pkl_name))
        if isinstance(meta, dict) and name in meta.get('arrays', {}):
            shape = tuple(meta['arrays'][name])
        break

    if shape is None:
        # No metadata: read as flat then treat as a single column.
        flat = np.memmap(dat_path, dtype='float64', mode='r')
        return np.asarray(flat).reshape(-1, 1)
    return np.memmap(dat_path, dtype='float64', mode='r', shape=shape)


def _accuracy_from_loss(classification_loss, axis):
    return 1.0 - np.asarray(classification_loss).mean(axis=axis)


def load_run(path, skipped_initial_val=False):
    """Load one run directory into per-run curves and scalars.

    ``skipped_initial_val`` handles runs recorded without the index-0 pre-training
    validation: the real validations fill rows 0..n_val-2 and the last row stays at
    its zero initialization. In that case we drop the trailing zero row (instead of
    the leading baseline) and shift the curve by one index so the last validation
    aligns with the end of training, matching the full-run layout.

    Returns a dict with keys:
        chip, seed, path,
        train: {x, target_error, classification_accuracy}
        val:   {x, target_error, classification_accuracy}
        test:  {target_error, classification_accuracy}      (scalars)
        best_val: {target_error, classification_accuracy}   (scalars)
    """
    chip, seed = _identify_run(path)
    par = _load_pickle(os.path.join(path, 'params.pkl'))
    batch_size = getattr(par, 'batch_size', 20) if par is not None else 20
    val_frequency = getattr(par, 'val_frequency', None) if par is not None else None

    run = {'chip': chip, 'seed': seed, 'path': path}

    # --- training curves ---
    tr_te = np.asarray(_memmap_metric(path, '', 'target_error'))
    tr_cl = np.asarray(_memmap_metric(path, '', 'classification_loss'))
    n_batches = tr_te.shape[0]
    run['train'] = {
        'x': (np.arange(1, n_batches + 1) * batch_size).astype(float),
        'target_error': tr_te.mean(axis=1),
        'classification_accuracy': _accuracy_from_loss(tr_cl, axis=1),
    }

    # --- validation curves (drop the pre-training baseline row 0) ---
    va_te = np.asarray(_memmap_metric(path, 'val_', 'target_error'))
    va_cl = np.asarray(_memmap_metric(path, 'val_', 'classification_loss'))
    if skipped_initial_val:
        # No initial validation recorded: real validations fill rows 0..n_val-2 and
        # the last row is leftover zero-init. Drop that trailing row and shift the
        # x-positions to start at the first post-step validation (start=1), so the
        # last real validation lands at the end of training like the full runs.
        va_te = va_te[:-1]
        va_cl = va_cl[:-1]
        start = 1
    else:
        start = 1 if DROP_VAL_BASELINE else 0
        va_te = va_te[start:]
        va_cl = va_cl[start:]
    n_val = va_te.shape[0]
    if val_frequency is None:
        # Infer the validation spacing from the training length.
        val_frequency = max(1, n_batches // (n_val + start))
    val_idx = np.arange(start, start + n_val)
    run['val'] = {
        'x': (val_idx * val_frequency * batch_size).astype(float),
        'target_error': va_te.mean(axis=1),
        'classification_accuracy': _accuracy_from_loss(va_cl, axis=1),
    }

    # --- test scalars ---
    te_te = np.asarray(_memmap_metric(path, 'test_', 'target_error'))
    te_cl = np.asarray(_memmap_metric(path, 'test_', 'classification_loss'))
    run['test'] = {
        'target_error': float(te_te.mean()),
        'classification_accuracy': float(1.0 - te_cl.mean()),
    }

    # --- best-validation scalars ---
    run['best_val'] = {
        'target_error': float(run['val']['target_error'].min()) if n_val else float('nan'),
        'classification_accuracy': float(run['val']['classification_accuracy'].max()) if n_val else float('nan'),
    }
    return run


def load_runs(paths, skipped_initial_val=False):
    """Load every run path; warn and skip any that fail to load."""
    runs = []
    for p in paths:
        try:
            runs.append(load_run(p, skipped_initial_val=skipped_initial_val))
        except (FileNotFoundError, ValueError) as e:
            warnings.warn(f"Skipping run '{p}': {e}")
    return runs


# ---------------------------------------------------------------------------
# Aggregation across runs in a group.
# ---------------------------------------------------------------------------
def _stack_curves(runs, split, metric, rolling_window):
    """Stack per-run 1D curves into (n_runs, L), truncating to the common min length.

    Training curves are smoothed with the rolling window; validation curves are left raw.
    Returns (x, stacked) or (None, None) if there are no runs.
    """
    if not runs:
        return None, None
    curves = []
    for r in runs:
        c = np.asarray(r[split][metric], dtype=float)
        if split == 'train' and rolling_window and rolling_window > 1:
            c, _ = calculate_rolling_mean(c, rolling_window)
        curves.append(c)
    lengths = [len(c) for c in curves]
    L = min(lengths)
    if len(set(lengths)) > 1:
        warnings.warn(
            f"{split}/{metric}: runs have differing lengths {sorted(set(lengths))}; "
            f"truncating to {L}."
        )
    stacked = np.stack([c[:L] for c in curves])
    x = np.asarray(runs[0][split]['x'], dtype=float)[:L]
    return x, stacked


def aggregate_group(runs, rolling_window=ROLLING_WINDOW):
    """Compute cross-run mean and std curves for every metric x {train, val}."""
    agg = {}
    for metric in METRICS:
        agg[metric] = {}
        for split in ('train', 'val'):
            x, stacked = _stack_curves(runs, split, metric, rolling_window)
            if stacked is None:
                agg[metric][split] = None
            else:
                agg[metric][split] = {
                    'x': x,
                    'mean': stacked.mean(axis=0),
                    'std': stacked.std(axis=0),
                    'n': stacked.shape[0],
                }
    return agg


# ---------------------------------------------------------------------------
# Scalar statistics and improvements.
# ---------------------------------------------------------------------------
def _scalar_stats(values):
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {'mean': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'max': float('nan'), 'n': 0}
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()), 'n': int(a.size)}


def compute_stats(runs):
    """Per metric x {test, best_val}: mean/std/min/max across runs."""
    stats = {}
    for metric in METRICS:
        stats[metric] = {}
        for eval_point in ('test', 'best_val'):
            stats[metric][eval_point] = _scalar_stats(
                [r[eval_point][metric] for r in runs]
            )
    return stats


def compute_improvements(frozen_runs, full_runs, warn=True):
    """Frozen->full improvement over runs sharing an exact ``(chip, seed)``.

    target error: frozen - full   (positive => full has lower error => better)
    accuracy:     full - frozen   (positive => full more accurate => better)
    """
    frozen_by_key = {(r['chip'], r['seed']): r for r in frozen_runs}
    full_by_key = {(r['chip'], r['seed']): r for r in full_runs}
    common = sorted(set(frozen_by_key) & set(full_by_key))

    result = {'pairs': common, 'metrics': {}}
    if not common and warn:
        warnings.warn(
            "No matching (chip, seed) pairs between frozen and full runs; "
            "improvement statistics are empty."
        )
    for metric, info in METRICS.items():
        result['metrics'][metric] = {}
        for eval_point in ('test', 'best_val'):
            diffs = []
            for key in common:
                f = frozen_by_key[key][eval_point][metric]
                u = full_by_key[key][eval_point][metric]
                diffs.append((f - u) if info['better'] == 'lower' else (u - f))
            result['metrics'][metric][eval_point] = _scalar_stats(diffs)
    return result


# ---------------------------------------------------------------------------
# Per-chip / per-seed breakdowns (reuse the reducers above on subsets).
# ---------------------------------------------------------------------------
def _group_by(runs, key):
    """Group runs by 'chip' or 'seed' -> {value: [runs]}, ordered by value."""
    groups = {}
    for r in runs:
        groups.setdefault(r[key], []).append(r)
    return {v: groups[v] for v in sorted(groups)}


def compute_grouped_stats(runs, key):
    """{group_value: compute_stats(subset)} for key in {'chip', 'seed'}."""
    return {v: compute_stats(sub) for v, sub in _group_by(runs, key).items()}


def compute_grouped_improvements(frozen_runs, full_runs, key):
    """{group_value: compute_improvements(frozen_subset, full_subset)} over all values present."""
    frozen_groups = _group_by(frozen_runs, key)
    full_groups = _group_by(full_runs, key)
    values = sorted(set(frozen_groups) | set(full_groups))
    return {
        v: compute_improvements(frozen_groups.get(v, []), full_groups.get(v, []), warn=False)
        for v in values
    }


# ---------------------------------------------------------------------------
# Console table.
# ---------------------------------------------------------------------------
def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    return f"{v:.{nd}f}"


_EVAL_NAMES = {'test': 'Test', 'best_val': 'Best validation'}


def _stats_rows(stats, group_name, extra=None):
    """Row dicts for a per-group stats result, with an optional leading column (e.g. {'Chip': 0})."""
    rows = []
    for metric, info in METRICS.items():
        for eval_point in ('test', 'best_val'):
            s = stats[metric][eval_point]
            row = dict(extra) if extra else {}
            row.update({
                'Metric': info['label'],
                'Eval': _EVAL_NAMES[eval_point],
                'Group': group_name,
                'n': s['n'],
                'Mean': _fmt(s['mean']),
                'Std': _fmt(s['std']),
                'Min': _fmt(s['min']),
                'Max': _fmt(s['max']),
            })
            rows.append(row)
    return rows


def _improvement_rows(improvements, extra=None):
    """Row dicts for a frozen->full improvement result, with an optional leading column."""
    rows = []
    for metric, info in METRICS.items():
        for eval_point in ('test', 'best_val'):
            s = improvements['metrics'][metric][eval_point]
            row = dict(extra) if extra else {}
            row.update({
                'Metric': info['label'],
                'Eval': _EVAL_NAMES[eval_point],
                'pairs': s['n'],
                'Mean improvement': _fmt(s['mean']),
                'Std': _fmt(s['std']),
                'Min': _fmt(s['min']),
                'Max': _fmt(s['max']),
            })
            rows.append(row)
    return rows


def _print_rows(title, rows):
    print(title)
    try:
        import pandas as pd
        print(pd.DataFrame(rows).to_string(index=False))
    except ImportError:
        _print_fixed_width(rows)


def print_table(frozen_stats, full_stats, improvements,
                frozen_runs, full_runs):
    """Print a formatted statistics table to the console."""
    rows = (_stats_rows(frozen_stats, 'Frozen') + _stats_rows(full_stats, 'Full'))
    rows.sort(key=lambda r: (r['Metric'], r['Eval'], r['Group']))
    imp_rows = _improvement_rows(improvements)

    header = (
        f"\n{'=' * 78}\n"
        f"Summary: {len(frozen_runs)} frozen run(s), {len(full_runs)} full run(s)\n"
        f"Improvement pairs (exact chip+seed match): {len(improvements['pairs'])}\n"
        f"{'=' * 78}"
    )
    print(header)

    _print_rows("\nPer-group statistics across runs:", rows)
    _print_rows("\nFrozen -> full improvement (positive = full is better):", imp_rows)
    print()


def print_grouped_table(key_label, grouped_frozen_stats, grouped_full_stats, grouped_improvements):
    """Print per-chip or per-seed stats and improvements (key_label e.g. 'Chip' or 'Seed')."""
    values = sorted(set(grouped_frozen_stats) | set(grouped_full_stats) | set(grouped_improvements))

    rows = []
    for v in values:
        if v in grouped_frozen_stats:
            rows += _stats_rows(grouped_frozen_stats[v], 'Frozen', extra={key_label: v})
        if v in grouped_full_stats:
            rows += _stats_rows(grouped_full_stats[v], 'Full', extra={key_label: v})
    rows.sort(key=lambda r: (r[key_label], r['Metric'], r['Eval'], r['Group']))

    imp_rows = []
    for v in values:
        if v in grouped_improvements:
            imp_rows += _improvement_rows(grouped_improvements[v], extra={key_label: v})
    imp_rows.sort(key=lambda r: (r[key_label], r['Metric'], r['Eval']))

    print(f"\n{'-' * 78}")
    _print_rows(f"Per-{key_label.lower()} statistics across runs:", rows)
    _print_rows(f"Per-{key_label.lower()} frozen -> full improvement (positive = full is better):",
                imp_rows)


def _print_fixed_width(rows):
    if not rows:
        print("(no data)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    line = '  '.join(f"{c:<{widths[c]}}" for c in cols)
    print(line)
    print('  '.join('-' * widths[c] for c in cols))
    for r in rows:
        print('  '.join(f"{str(r[c]):<{widths[c]}}" for c in cols))


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def plot_metric(metric, frozen_agg, full_agg, rolling_window=ROLLING_WINDOW, savepath=None):
    """One figure for ``metric`` with four mean +/- std lines.

    full-train (solid blue), full-validation (solid purple),
    frozen-train (dashed blue), frozen-validation (dashed purple).
    Returns the Figure. If ``savepath`` is given, also saves PDF (and PNG alongside).
    """
    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)

    # (group label, aggregate, fill alpha, per-split colours)
    series = [
        ('Full', full_agg, 0.20, {'train': COLOR_TRAIN, 'val': COLOR_VAL}),
        ('Frozen', frozen_agg, 0.15, {'train': COLOR_TRAIN_LIGHT, 'val': COLOR_VAL_LIGHT}),
    ]
    split_labels = {'train': 'train', 'val': 'validation'}

    plotted = False
    for group_label, agg, alpha, colors in series:
        if agg is None:
            continue
        for split, split_label in split_labels.items():
            d = agg[metric][split]
            if d is None:
                continue
            color = colors[split]
            label = f"{group_label} {split_label} (n={d['n']})"
            ax.plot(d['x'], d['mean'], '-', color=color, linewidth=1.6, label=label)
            ax.fill_between(d['x'], d['mean'] - d['std'], d['mean'] + d['std'],
                            color=color, alpha=alpha, linewidth=0)
            plotted = True

    ax.set_xlabel('# examples')
    ax.set_ylabel(METRICS[metric]['label'])
    title = METRICS[metric]['label'].split(' [')[0]
    if rolling_window and rolling_window > 1:
        title += f" (train rolling window = {rolling_window})"
    ax.set_title(title)
    ax.set_yscale(METRICS[metric].get('yscale', 'linear'))
    ax.set_ylim(*METRICS[metric].get('ylim', (None, None)))
    ax.grid(True, which='both')
    if plotted:
        ax.legend()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
        fig.savefig(savepath + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
    return fig


def plot_grouped_bars(grouped_frozen_stats, grouped_full_stats, key_label, savepath=None,
                      eval_point='best_val'):
    """Grouped Full vs Frozen bar charts broken down per chip or per seed.

    One panel per metric (target error, accuracy), each showing grouped bars over the key values
    (chips/seeds) at a single evaluation point (best validation by default; test is still reported
    in the statistics tables). Bar height = subgroup mean, error bar = subgroup std. Reads the
    pre-computed grouped stats; performs no new statistics.
    """
    values = sorted(set(grouped_frozen_stats) | set(grouped_full_stats))
    x = np.arange(len(values))
    tick_labels = [f"{key_label.lower()} {v}" for v in values]
    # Long labels (e.g. seeds) crowd each other -> rotate them.
    rotation = 25 if max((len(t) for t in tick_labels), default=0) > 7 else 0
    width = 0.38
    metrics = list(METRICS)

    latexify(width_fraction=0.95, base_font_size=9, height_ratio=0.5)
    fig, axes = plt.subplots(1, len(metrics), constrained_layout=True, squeeze=False)

    def _mean_std(grouped, v, metric):
        if v not in grouped:
            return np.nan, np.nan
        s = grouped[v][metric][eval_point]
        return s['mean'], (s['std'] if not np.isnan(s['std']) else 0.0)

    for j, metric in enumerate(metrics):
        ax = axes[0][j]
        full_m = [_mean_std(grouped_full_stats, v, metric)[0] for v in values]
        full_e = [_mean_std(grouped_full_stats, v, metric)[1] for v in values]
        froz_m = [_mean_std(grouped_frozen_stats, v, metric)[0] for v in values]
        froz_e = [_mean_std(grouped_frozen_stats, v, metric)[1] for v in values]

        ax.bar(x - width / 2, full_m, width, yerr=full_e, capsize=3,
               color=COLOR_TRAIN, label='Full', error_kw={'elinewidth': 1})
        ax.bar(x + width / 2, froz_m, width, yerr=froz_e, capsize=3,
               color=COLOR_TRAIN_LIGHT, label='Frozen', error_kw={'elinewidth': 1})

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=rotation,
                           ha='right' if rotation else 'center')
        ax.set_ylabel(METRICS[metric]['label'])
        # Bars stay on a linear axis (a log axis has no valid 0 baseline for bars);
        # zoom the y-range past 0 so the small between-group differences are legible.
        ax.set_ylim(*METRICS[metric].get('bar_ylim', (None, None)))
        ax.grid(True, axis='y', alpha=0.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    fig.suptitle(f"Per-{key_label.lower()} comparison at {_EVAL_NAMES[eval_point].lower()} "
                 f"(error bars = std across runs)")

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
        fig.savefig(savepath + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
    return fig


def export_for_typst(full_agg, frozen_agg, filepath="data.json"):
    """Export aggregated training and validation accuracy for both regimes to JSON."""
    
    def extract_curves(agg):
        if agg is None:
            return {"train": [], "val": []}
        acc = agg['classification_accuracy']
        return {
            "train": [
                [float(x), float(m), float(s)] 
                for x, m, s in zip(acc['train']['x'], acc['train']['mean'], acc['train']['std'])
            ],
            "val": [
                [float(x), float(m), float(s)] 
                for x, m, s in zip(acc['val']['x'], acc['val']['mean'], acc['val']['std'])
            ]
        }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump({
            "full": extract_curves(full_agg),
            "frozen": extract_curves(frozen_agg)
        }, f)


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------
def summarize(frozen_paths, full_paths, rolling_window=ROLLING_WINDOW,
              output_dir=OUTPUT_DIR, save=True, show=False,
              frozen_skipped_initial_val=False):
    """Load, compute statistics, print the table, and build both plots.

    Set ``frozen_skipped_initial_val=True`` when the frozen runs were recorded
    without the index-0 pre-training validation (see ``load_run``).

    Returns a dict with the loaded runs, aggregates, stats, improvements and figures.
    """
    frozen_runs = load_runs(frozen_paths, skipped_initial_val=frozen_skipped_initial_val)
    full_runs = load_runs(full_paths)
    if not frozen_runs and not full_runs:
        raise ValueError("No runs could be loaded. Check FROZEN_PATHS / FULL_PATHS.")

    frozen_agg = aggregate_group(frozen_runs, rolling_window) if frozen_runs else None
    full_agg = aggregate_group(full_runs, rolling_window) if full_runs else None
    frozen_stats = compute_stats(frozen_runs)
    full_stats = compute_stats(full_runs)
    improvements = compute_improvements(frozen_runs, full_runs)

    if frozen_agg is not None and full_agg is not None:
        # This will save data.json in the same folder as your plots
        export_for_typst(full_agg, frozen_agg, filepath=os.path.join(output_dir, "dynapse_training_curves.json"))

    print_table(frozen_stats, full_stats, improvements, frozen_runs, full_runs)

    # Per-chip and per-seed breakdowns.
    grouped = {}
    for key, label in (('chip', 'Chip'), ('seed', 'Seed')):
        gf = compute_grouped_stats(frozen_runs, key)
        gu = compute_grouped_stats(full_runs, key)
        gi = compute_grouped_improvements(frozen_runs, full_runs, key)
        grouped[key] = {'frozen': gf, 'full': gu, 'improvements': gi, 'label': label}
        print_grouped_table(label, gf, gu, gi)

    figures = {}
    for metric in METRICS:
        savepath = os.path.join(output_dir, f"summary_{metric}") if save else None
        figures[metric] = plot_metric(metric, frozen_agg, full_agg, rolling_window, savepath)
        if save:
            print(f"Saved plot: {savepath}.pdf / .png")

    for key in ('chip', 'seed'):
        savepath = os.path.join(output_dir, f"summary_bars_by_{key}") if save else None
        figures[f'bars_by_{key}'] = plot_grouped_bars(
            grouped[key]['frozen'], grouped[key]['full'], grouped[key]['label'], savepath)
        if save:
            print(f"Saved plot: {savepath}.pdf / .png")

    if show:
        plt.show()

    return {
        'frozen_runs': frozen_runs, 'full_runs': full_runs,
        'frozen_agg': frozen_agg, 'full_agg': full_agg,
        'frozen_stats': frozen_stats, 'full_stats': full_stats,
        'improvements': improvements, 'figures': figures,
        'frozen_stats_by_chip': grouped['chip']['frozen'],
        'full_stats_by_chip': grouped['chip']['full'],
        'improvements_by_chip': grouped['chip']['improvements'],
        'frozen_stats_by_seed': grouped['seed']['frozen'],
        'full_stats_by_seed': grouped['seed']['full'],
        'improvements_by_seed': grouped['seed']['improvements'],
    }


def _expand(paths):
    """Allow glob patterns in the configured path lists."""
    out = []
    for p in paths:
        matches = sorted(glob.glob(p))
        out.extend(matches if matches else [p])
    return out


def main():
    frozen_paths = _expand(FROZEN_PATHS)
    full_paths = _expand(FULL_PATHS)
    if not frozen_paths and not full_paths:
        raise SystemExit(
            "Configure FROZEN_PATHS and FULL_PATHS at the top of summarize_runs.py "
            "(or pass them to summarize())."
        )
    summarize(frozen_paths, full_paths, rolling_window=ROLLING_WINDOW,
              output_dir=OUTPUT_DIR, save=True, show=False)


if __name__ == '__main__':
    main()
