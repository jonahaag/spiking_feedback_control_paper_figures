"""Shared color scheme and plotting helpers for scripts/create_figures.ipynb.

Color encodes training regime, linestyle encodes train/val, so the scheme stays consistent and
legible across all figures added to the notebook.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from summarize_runs import latexify

# Paul Tol's colorblind-safe muted palette.
COLOR_FULL = '#117733'        # green -- full training regime (2b/3b, 4b/4c)
COLOR_FROZEN = '#AA4499'      # purple -- frozen training regime (2b/3b, 4b/4c)
COLOR_REFERENCE = '#999933'   # olive, reserved for reference/baseline lines
COLOR_REFERENCE_2 = '#DDCC77' # sand, reserved for second reference/baseline lines

COLOR_KP = '#CC6677'          # rose -- KP feedback regime (currently not plotted)
COLOR_TARGET = '#CC6677'      # rose (3f, reused for predicted-class 2f)
COLOR_NONTARGET = '#44AA99'   # teal (3f, reused for predicted-class 2f)
COLOR_HIDDEN = '#332288'      # indigo -- hidden features (2d/4d) and input-to-hidden weights (2e)
COLOR_HIDDEN_OUT = '#88CCEE'  # cyan -- hidden-to-output weights (2e)
COLOR_FEEDBACK = '#882255'    # wine -- feedback weights (2e)

CMAP_HIDDEN = LinearSegmentedColormap.from_list('hidden', [(1, 1, 1, 1), to_rgba(COLOR_HIDDEN)], N=100)  # (2d/4d)

# Training regimes, in the order they appear in every figure (legends, x positions). 'full' is
# the learnt-feedback regime (Rec-LRA), 'fixed-feedback' fixed random feedback (DFA).
# REGIME_LABEL is the single place that sets how the regimes are named in all figures.
REGIME_COLOR = {'frozen': COLOR_FROZEN, 'fixed-feedback': COLOR_NONTARGET, 'full': COLOR_FULL}
REGIME_LABEL = {'frozen': 'Frozen', 'fixed-feedback': 'Fixed Feedback', 'full': 'Full Local'}

# Simulation runs label regimes by their wandb 'note'.
NOTE_REGIME = {'frozen-hidden': 'frozen', 'fixed-feedback': 'fixed-feedback', 'locally-learnt': 'full'}
SPLIT_LABEL = {'train': 'training', 'val': 'validation'}
SPLIT_LINE_ALPHA = {'train': 0.5, 'val': 1.0}
TRAIN_FILL_ALPHA = 0.18


# --- shared style helpers ---------------------------------------------------------------------

def _sparse_grid(ax, axis='y', alpha=0.35):
    """Grid at fewer, more widely-spaced ticks rather than matplotlib's dense default."""
    if axis in ('both', 'y'):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if axis in ('both', 'x'):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, axis='y', alpha=alpha)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- accuracy over training steps (2b/3b) -------------------------------------------------------

def _downsample_mean(df, bin_size):
    """Non-overlapping block-average of consecutive rows within each (regime, seed, chip) run.

    Unlike a backward-looking rolling mean, a block average is centered within its block, so it
    doesn't lag behind the true curve -- it only reduces the number of plotted points.
    """
    df = df.sort_values('step').copy()
    df['bin'] = df.groupby(['regime', 'seed', 'chip']).cumcount() // bin_size
    return (
        df.groupby(['regime', 'seed', 'chip', 'bin'], as_index=False)
        .agg(step=('step', 'mean'), accuracy=('accuracy', 'mean'))
        .drop(columns='bin')
    )


def aggregate_mean_std(df, train_bin_size=90):
    """Per (regime, split, step): mean (center) +/- std (lower/upper) across seeds x chips.

    Train curves are downsampled per-run (block-averaged over ``train_bin_size`` consecutive
    batches) before aggregating; validation curves are left as recorded.
    """
    train_ds = _downsample_mean(df[df['split'] == 'train'], train_bin_size)
    train_ds['split'] = 'train'
    df = pd.concat([train_ds, df[df['split'] != 'train']], ignore_index=True)
    agg = (
        df.groupby(['regime', 'split', 'step'])['accuracy']
        .agg(center='mean', std='std', n='count')
        .reset_index()
    )
    agg['lower'] = agg['center'] - agg['std']
    agg['upper'] = agg['center'] + agg['std']
    return agg


def _with_chance_point(d, chance_level):
    """Prepend an untrained-network anchor point at step 0, accuracy = chance_level."""
    anchor = pd.DataFrame([{'step': 0, 'center': chance_level, 'lower': chance_level, 'upper': chance_level}])
    return pd.concat([anchor, d], ignore_index=True).sort_values('step')


def plot_accuracy_over_steps(agg, ax=None, reference_lines=None, chance_level=None,
                             regimes=('frozen', 'full')):
    """Accuracy curves (center +/- band, typically from ``aggregate_mean_std``): color = regime.
    Training is a solid line with a shaded +/- band; validation is scattered markers only (mean,
    no error bar/band). Only ``regimes`` are drawn (2b/3b: Frozen vs. Rec-LRA), in
    ``REGIME_COLOR`` order.

    ``chance_level``, if given, prepends a step-0 point (untrained network) to both curves.
    """
    if ax is None:
        latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
        _, ax = plt.subplots(constrained_layout=True)

    for regime, color in REGIME_COLOR.items():
        if regime not in regimes:
            continue
        for split in ('train', 'val'):
            d = agg[(agg['regime'] == regime) & (agg['split'] == split)]
            if d.empty:
                continue
            if chance_level is not None:
                d = _with_chance_point(d, chance_level)
            label = f"{REGIME_LABEL[regime]} {SPLIT_LABEL[split]}"
            if split == 'val':
                ax.scatter(d['step'], d['center'], color=color, s=18, linewidths=0,
                          alpha=SPLIT_LINE_ALPHA[split], label=label, zorder=3)
            else:
                ax.plot(d['step'], d['center'], color=color, linewidth=1.6,
                        alpha=SPLIT_LINE_ALPHA[split], label=label)
                ax.fill_between(d['step'], d['lower'], d['upper'], color=color,
                                alpha=TRAIN_FILL_ALPHA, linewidth=0)

    for label, y in (reference_lines or {}).items():
        ax.axhline(y, color=COLOR_REFERENCE, linestyle=':', linewidth=1.4, label=label)

    ax.set_xlabel('Training steps')
    ax.set_ylabel('Classification accuracy')
    _sparse_grid(ax)
    ax.legend()
    return ax


# --- feedback current (3f) ----------------------------------------------------------------------

def aggregate_binned(df, value_col, group_cols=(), bin_size=60):
    """Bin consecutive steps into groups of ``bin_size`` and compute the mean of ``value_col``.

    Uses the mean (not median) because the underlying rates are quantized (derived from spike
    counts), so the median snaps between a handful of discrete values and looks like a staircase
    -- the mean varies continuously. ``group_cols`` (e.g. ``['class']``) are kept as separate
    series -- the step binning is shared across groups since they cover the same steps.
    """
    df = df.sort_values('step').copy()
    steps = np.sort(df['step'].unique())
    bin_of_step = pd.Series(np.arange(len(steps)) // bin_size, index=steps)
    df['bin'] = bin_of_step.loc[df['step']].to_numpy()
    return (
        df.groupby([*group_cols, 'bin'], as_index=False)
        .agg(step=('step', 'mean'), **{value_col: (value_col, 'mean')})
        .drop(columns='bin')
    )


def plot_feedback_current(agg_target, agg_nontarget):
    """Measured output rate per output class, split into target- vs. non-target-class samples.

    Low-alpha per-class traces (one per output neuron) with a bold overall mean overlaid, for
    each group.
    """
    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)

    groups = [
        ('Target class', agg_target, 'target_rate', COLOR_TARGET),
        ('Non-target classes', agg_nontarget, 'nontarget_rate', COLOR_NONTARGET),
    ]
    for label, agg, value_col, color in groups:
        for _, trace in agg.groupby('class'):
            ax.plot(trace['step'], trace[value_col], color=color, linestyle=(0, (2, 1.5)),
                    linewidth=1.6, alpha=0.3)
        mean = agg.groupby('step', as_index=False)[value_col].mean()
        ax.plot(mean['step'], mean[value_col], color=color, linewidth=2.2, label=label)

    ax.set_xlabel('Training steps')
    ax.set_ylabel('Output rate [Hz]')
    _sparse_grid(ax)
    ax.legend()

    return fig


# --- hidden feature maps (2d/4d) ------------------------------------------------------------------

def _draw_hidden_row(axes_row, d, neuron_indices, vmin, vmax, lims):
    """One row of scatter panels (one per neuron in ``neuron_indices``) for a single (chip, step)
    subset of hidden activity."""
    for ax, neuron in zip(axes_row, neuron_indices):
        values = d[f'hidden_{neuron}'].to_numpy()
        ax.scatter(d['x'], d['y'], c=values, cmap=CMAP_HIDDEN, vmin=vmin, vmax=vmax,
                  alpha=1.0, s=16, linewidths=0)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def select_most_changed_neurons(df, init_step, final_step, n_hidden=15, max_neurons=5):
    """Rank hidden neurons by how much their overall firing rate changed between ``init_step``
    and ``final_step`` (absolute change in mean rate over the validation samples), and return the
    top ``max_neurons`` indices, most-changed first.

    Mean-rate change (rather than e.g. the std of the per-sample difference) is what reads as an
    "obvious" change in the plotted feature map -- a neuron whose spatial pattern is reshuffled
    but keeps a similar overall intensity looks similar at a glance in both panels.
    """
    hidden_cols = [f'hidden_{i}' for i in range(n_hidden)]
    init = df[df['step'] == init_step].sort_values('sample')[hidden_cols].to_numpy()
    final = df[df['step'] == final_step].sort_values('sample')[hidden_cols].to_numpy()
    change = np.abs(final.mean(axis=0) - init.mean(axis=0))
    order = np.argsort(change)[::-1]
    return order[:max_neurons].tolist()


def plot_hidden_feature_maps(df, n_hidden=15, neuron_indices=None, input_frmin=10, input_frmax=100):
    """n_steps x n_neurons grid: each panel is a scatter of validation samples over (x, y) input
    space, colored by one hidden neuron's firing rate (white = low, COLOR_FULL = high). Rows are
    training steps (``df['step']``, ascending); columns are ``neuron_indices`` (default: all
    ``n_hidden`` neurons). One shared color scale across all panels so amplitude differences
    between neurons and between steps stay visible.
    """
    if neuron_indices is None:
        neuron_indices = range(n_hidden)
    hidden_cols = [f'hidden_{i}' for i in neuron_indices]
    vmin = df[hidden_cols].to_numpy().min()
    vmax = df[hidden_cols].to_numpy().max()
    lims = (input_frmin - 6, input_frmax + 6)

    steps = sorted(df['step'].unique())
    n_rows = len(steps)
    n_cols = len(hidden_cols)

    latexify(width_fraction=1.0, base_font_size=7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.85, n_rows * 1.15),
                             constrained_layout=True, squeeze=False)

    for row, step in enumerate(steps):
        _draw_hidden_row(axes[row], df[df['step'] == step], neuron_indices, vmin, vmax, lims)

    return fig


def plot_hidden_feature_maps_across_chips(df, chips, init_chip, init_step, final_step,
                                          n_hidden=15, input_frmin=10, input_frmax=100):
    """Like plot_hidden_feature_maps, but contrasts a shared initialization (row 0) against the
    final-training hidden activity of each chip (one row per chip) -- all chips share the same
    weight_seed, so ``init_step`` on ``init_chip`` is representative of every chip's starting
    point.
    """
    neuron_indices = range(n_hidden)
    hidden_cols = [f'hidden_{i}' for i in neuron_indices]
    vmin = df[hidden_cols].to_numpy().min()
    vmax = df[hidden_cols].to_numpy().max()
    lims = (input_frmin - 6, input_frmax + 6)

    row_specs = [df[(df['chip'] == init_chip) & (df['step'] == init_step)]]
    for chip in chips:
        row_specs.append(df[(df['chip'] == chip) & (df['step'] == final_step)])

    latexify(width_fraction=1.0, base_font_size=7)
    fig, axes = plt.subplots(len(row_specs), n_hidden,
                             figsize=(n_hidden * 0.85, len(row_specs) * 1.15), constrained_layout=True)

    for row, d in enumerate(row_specs):
        _draw_hidden_row(axes[row], d, neuron_indices, vmin, vmax, lims)

    return fig


# --- grouped box plots (4b/4c) --------------------------------------------------------------------

def compute_best_val(df, value_col, better, by=('regime', 'seed', 'chip')):
    """Best validation value (max if better='higher', min if better='lower') per run (``by``)."""
    val = df[df['split'] == 'val']
    agg = 'max' if better == 'higher' else 'min'
    return val.groupby(list(by), as_index=False)[value_col].agg(agg)


def _lighten(color_hex, factor=0.45):
    """Blend ``color_hex`` towards white (simulation boxes in 2c/3c)."""
    from matplotlib.colors import to_hex, to_rgb
    rgb = np.array(to_rgb(color_hex))
    return to_hex((1 - factor) * rgb + factor * 1.0)


def _boxplot(ax, data, positions, width, color):
    """Filled box plot in ``color`` with a black median (shared by 2c/3c and 4b/4c)."""
    ax.boxplot(
        data, positions=positions, widths=width, patch_artist=True,
        boxprops=dict(facecolor=color, color=color),
        medianprops=dict(color='black'),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2),
    )


def _plot_grouped_box(df, group_col, tick_label, legend_loc='best', legend_ncol=1, xlabel=None):
    """Best validation accuracy per regime, grouped by ``group_col`` (x axis), as box plots over the
    remaining runs. Only regimes present in ``df`` are plotted, in ``REGIME_COLOR`` order.
    ``tick_label`` formats a group value, e.g. ``'Chip {}'``.
    """
    from matplotlib.patches import Patch

    by = ('regime', 'seed', 'chip') if group_col == 'chip' else ('regime', group_col, 'seed', 'chip')
    best = compute_best_val(df, 'accuracy', 'higher', by=by)
    regimes = [r for r in REGIME_COLOR if r in best['regime'].unique()]

    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)
    width = 0.8 / len(regimes)

    values = sorted(best[group_col].unique())
    x = np.arange(len(values))
    offsets = (np.arange(len(regimes)) - (len(regimes) - 1) / 2) * width
    for regime, offset in zip(regimes, offsets):
        data, positions = [], []
        for i, value in enumerate(values):
            d = best[(best['regime'] == regime) & (best[group_col] == value)]['accuracy'].dropna().values
            if len(d) > 0:
                data.append(d)
                positions.append(x[i] + offset)
        if data:
            _boxplot(ax, data, positions, width * 0.8, REGIME_COLOR[regime])

    ax.set_xticks(x)
    ax.set_xticklabels([tick_label.format(v) for v in values])
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Best validation accuracy')
    span = best['accuracy'].max() - best['accuracy'].min()
    ax.set_ylim(max(0, best['accuracy'].min() - 0.3 * span), best['accuracy'].max() + 0.3 * span)
    _sparse_grid(ax, axis='y')
    ax.legend(handles=[Patch(facecolor=REGIME_COLOR[r], label=REGIME_LABEL[r]) for r in regimes], loc=legend_loc,
              ncol=legend_ncol)

    return fig


def plot_grouped_box_by_chip(df, legend_loc='best', legend_ncol=1):
    """Best validation accuracy per regime, grouped by chip (4b/4c): shows whether results are
    consistent across chips."""
    return _plot_grouped_box(df, 'chip', 'Chip {}', legend_loc=legend_loc, legend_ncol=legend_ncol)


# --- population size ------------------------------------------------------------------------------

def plot_box_by_population(df, legend_loc='best', legend_ncol=1):
    """Best validation accuracy per regime vs. population size (``n_pop`` column), as box plots
    over seeds x chips, in the 4b/4c style."""
    return _plot_grouped_box(df, 'n_pop', '{}', legend_loc=legend_loc, legend_ncol=legend_ncol,
                             xlabel='Population size $n_{pop}$')


def plot_population_accuracy_over_steps(df, n_pop, chance_level=0.1):
    """3b-style accuracy over training for a single population size: training line +/- band,
    validation markers, and the mean test accuracy marked at the end of training (no error bar).
    """
    d = df[df['n_pop'] == n_pop]
    agg = aggregate_mean_std(d[d['split'] != 'test'])
    regimes = [r for r in REGIME_COLOR if r in agg['regime'].unique()]
    ax = plot_accuracy_over_steps(agg, chance_level=chance_level, regimes=regimes)

    test = d[d['split'] == 'test'].groupby('regime').agg(step=('step', 'max'), accuracy=('accuracy', 'mean'))
    for regime in regimes:
        if regime in test.index:
            t = test.loc[regime]
            ax.scatter(t['step'], t['accuracy'], color=REGIME_COLOR[regime], marker='D', s=24,
                       edgecolors='black', linewidths=0.6, zorder=4, label=f"{REGIME_LABEL[regime]} test")
    ax.legend()
    return ax.figure


# --- weight distributions (2e/3d) -----------------------------------------------------------------

# Reused across datasets: color/title assigned by subplot position (earliest -> latest weight
# matrix), not by semantic weight name, so different weight-graph shapes (e.g. yin-yang's single
# hidden layer + feedback vs. mnist1d's two hidden layers) can share the same visual identity.
YINYANG_WEIGHT_COLOR = {'in_hidden': COLOR_HIDDEN, 'hidden_out': COLOR_HIDDEN_OUT, 'feedback': COLOR_FEEDBACK}
YINYANG_WEIGHT_TITLE = {'in_hidden': 'Input → hidden', 'hidden_out': 'Hidden → output', 'feedback': 'Feedback'}

MNIST1D_WEIGHT_COLOR = {
    'in_hid1': COLOR_HIDDEN,
    'hid1_hid2': COLOR_HIDDEN_OUT,
    'hid2_out': COLOR_NONTARGET,
    'fb_out_hid1': COLOR_FEEDBACK,
    'fb_out_hid2': COLOR_FEEDBACK,
}
MNIST1D_WEIGHT_TITLE = {
    'in_hid1': 'Input → hidden 1',
    'hid1_hid2': 'Hidden 1 → hidden 2',
    'hid2_out': 'Hidden 2 → output',
    'fb_out_hid1': 'Feedback → hidden 1',
    'fb_out_hid2': 'Feedback → hidden 2',
}

PHASE_LABEL = {'init': 'Initialization', 'end': 'End of training'}
PHASE_FILL_ALPHA = {'init': 0.25, 'end': 0.45}
PHASE_LINESTYLE = {'init': ':', 'end': '-'}


def plot_weight_distributions(df, colors, titles, pad_fraction=0.15, bw_adjust=0.6):
    """1xN grid: KDE-smoothed weight distribution at initialization vs. end of training, one
    panel per weight type (``colors``/``titles`` keys, in order), each in its own color. The area
    where the two distributions overlap is hatched to make it easy to spot.

    ``bw_adjust`` scales the KDE bandwidth relative to scipy's default (Scott's rule); less than
    1 makes the curve track the data more closely (less smoothing).
    """
    weight_types = [w for w in colors if w in set(df['weight_type'])]
    n_cols = len(weight_types)
    latexify(width_fraction=n_cols / 3.0, base_font_size=9, height_ratio=0.4 * 3.0 / n_cols)
    fig, axes = plt.subplots(1, n_cols, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (ax, weight_type) in enumerate(zip(axes, weight_types)):
        color = colors[weight_type]
        d = df[df['weight_type'] == weight_type]
        w_min, w_max = d['weight'].min(), d['weight'].max()
        pad = pad_fraction * (w_max - w_min)
        x = np.linspace(w_min - pad, w_max + pad, 200)

        densities = {}
        for phase in ('init', 'end'):
            values = d.loc[d['phase'] == phase, 'weight']
            kde = gaussian_kde(values)
            kde.set_bandwidth(kde.factor * bw_adjust)
            density = kde(x)
            densities[phase] = density
            ax.plot(x, density, color=color, linestyle=PHASE_LINESTYLE[phase], linewidth=1.6,
                   label=PHASE_LABEL[phase])
            ax.fill_between(x, density, color=color, alpha=PHASE_FILL_ALPHA[phase], linewidth=0)

        overlap = np.minimum(densities['init'], densities['end'])
        ax.fill_between(x, overlap, facecolor='none', edgecolor='black', hatch='///',
                        linewidth=0, alpha=0.5, label='Overlap' if i == 0 else None)

        ax.set_title(titles[weight_type])
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel('Weight')
        ax.tick_params(axis='x', labelsize=8)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)

    return fig


# --- predictions / ground truth feature maps (2a/2f) -----------------------------------------------

CLASS_COLOR = [COLOR_TARGET, COLOR_HIDDEN, COLOR_NONTARGET]  # rose / indigo / teal (2a/2f)


def plot_predictions(df, class_col='predicted_class', colors=CLASS_COLOR, input_frmin=10, input_frmax=100):
    """Feature map over (x, y) input space, colored by ``class_col`` (predicted or true class)."""
    lims = (input_frmin - 6, input_frmax + 6)

    latexify(width_fraction=0.5, base_font_size=9)
    fig, ax = plt.subplots(constrained_layout=True)

    for cls, color in enumerate(colors):
        d = df[df[class_col] == cls]
        ax.scatter(d['x'], d['y'], color=color, s=16, linewidths=0)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal')
    ax.set_xticks([input_frmin, input_frmax])
    ax.set_yticks([input_frmin, input_frmax])
    ax.set_xlabel('Input x [Hz]')
    ax.set_ylabel('Input y [Hz]')
    ax.grid(True, color='black', alpha=1)

    return fig

def plot_sim_vs_dynapse_box_plots(df_dynapse, df_sim, y_min=0.0, y_max=1.0, group_order=None,
                                  hide_sim_for=(), legend_loc='upper left'):
    """Box plot comparing Simulation vs. Dynapse for various methods (2c/3c).

    Expects df_sim to contain 'note' and 'test accuracy (Max)'
    Expects df_dynapse to contain 'regime' and 'accuracy' (processed via compute_best_val)

    ``hide_sim_for`` lists notes whose simulation box is suppressed (only the Dynapse box is
    drawn, centered). Training regimes use the shared ``REGIME_COLOR``/``REGIME_LABEL``.
    """
    from matplotlib.patches import Patch

    hide_sim_for = set(hide_sim_for)
    if group_order is None:
        group_order = ['frozen-hidden', 'fixed-feedback', 'locally-learnt', 'surrogate gradient', 'backprop']

    # Process Dynapse
    best_dynapse = compute_best_val(df_dynapse, 'accuracy', 'higher')
    regime_note = {r: n for n, r in NOTE_REGIME.items()}
    if 'note' not in best_dynapse.columns:
        best_dynapse['note'] = best_dynapse['regime'].map(regime_note).fillna(best_dynapse['regime'])

    # Style matching plot_grouped_box_by_chip
    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)
    width = 0.38

    # Method color mapping using Paul Tol palette defined at top of file
    method_color_map = {
        **{n: REGIME_COLOR[r] for n, r in NOTE_REGIME.items()},
        'kp': COLOR_KP,
        'surrogate gradient': COLOR_HIDDEN_OUT,
        'backprop': COLOR_REFERENCE,
    }

    x = np.arange(len(group_order))
    for i, note in enumerate(group_order):
        d_data = best_dynapse[best_dynapse['note'] == note]['accuracy'].dropna().values
        s_data = df_sim[df_sim['note'] == note]['test accuracy (Max)'].dropna().values
        if note in hide_sim_for:
            s_data = s_data[:0]

        c_dyn = method_color_map.get(note, COLOR_FULL)
        c_sim = _lighten(c_dyn)

        if len(d_data) > 0 and len(s_data) > 0:
            # Both present -> side-by-side simulation / Dynapse boxes.
            _boxplot(ax, [s_data], [x[i] - width / 2], width * 0.8, c_sim)
            _boxplot(ax, [d_data], [x[i] + width / 2], width * 0.8, c_dyn)
        elif len(d_data) > 0:
            # Dynapse only -> single centered box.
            _boxplot(ax, [d_data], [x[i]], width * 0.8, c_dyn)
        elif len(s_data) > 0:
            # Simulation only (surrogate gradient / backprop) -> single centered box.
            _boxplot(ax, [s_data], [x[i]], width * 0.8, c_sim)

    ax.set_xticks(x)
    label_map = {
        **{n: REGIME_LABEL[r].replace(' ', '\n') for n, r in NOTE_REGIME.items()},  # wrap long tick labels
        'kp': 'KP',
        'surrogate gradient': 'Surrogate\nGradient',
        'backprop': 'ANN\n(SGD)'
    }
    tick_labels = [label_map.get(note, str(note).replace('-', ' ').title()) for note in group_order]
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')
    ax.set_ylabel('Best validation accuracy')
    ax.tick_params(axis='both', labelsize=8)

    ax.set_ylim(y_min, y_max)
    _sparse_grid(ax, axis='y')

    legend_elements = [
        Patch(facecolor=_lighten('#555555'), label='Simulation'),
        Patch(facecolor='#555555', label='Dynapse')
    ]
    ax.legend(handles=legend_elements, loc=legend_loc)

    return fig


# --- approaching hardware box plots ---------------------------------------------------------------

HW_APPROACH_ORDER = (
    'baseline',
    'dpi_sat',
    'w_sat',
    'dpi_sat_w_sat',
    'w_clip',
    'dpi_sat_wclip',
    'dpi_sat_ei_asym_quant',
    'vfloor',
    'ei_asym',
    'ei_asym_vfloor',
    'ei_asym_wclip',
    'dpi_sat_ei_asym_wclip',
    'quantized',
    'full_hw_noquant',
    'full_hw_noquant_init_tuned',
    'full_hw_quant',
)

HW_APPROACH_LABEL = {
    'baseline': 'Software\nbaseline',
    'dpi_sat': 'DPI sat.',
    'w_sat': 'Weight sat.',
    'dpi_sat_w_sat': 'DPI +\nweight sat.',
    'dpi_sat_ei_asym': 'DPI sat. +\nE/I asym.',
    'dpi_sat_ei_asym_quant': 'DPI sat. +\nE/I asym. +\nweight quant.',
    'w_clip': 'Weight clip',
    'dpi_sat_wclip': 'DPI sat. +\nweight clip',
    'vfloor': '$V_{\\mathrm{floor}}$',
    'ei_asym': 'E/I asym.',
    'ei_asym_vfloor': 'E/I asym. +\n$V_{\\mathrm{floor}}$',
    'ei_asym_wclip': 'E/I asym. +\nweight clip',
    'dpi_sat_ei_asym_wclip': 'DPI sat. + E/I\nasym. + weight clip',
    'quantized': 'Quantized',
    'full_hw_noquant': 'Full HW\n(no quant.)',
    'full_hw_noquant_init_tuned': 'Full HW\n(tuned init.)',
    'full_hw_quant': 'Full HW\n(quant.)',
    'dynapse': 'Dynapse',
}


def _parse_hw_approach_note(note):
    """Split a simulation ``note`` into ``(hw_config, regime)``.

    The last two hyphen-separated tokens identify the learning algorithm
    (``frozen-hidden`` -> ``frozen``, ``fixed-feedback`` -> ``fixed-feedback``,
    ``locally-learnt`` -> ``full``); the preceding prefix identifies the modeled hardware
    non-ideality, or ``'baseline'`` when only the learning algorithm is present.
    """
    if not isinstance(note, str):
        return None, None
    parts = note.strip().split('-')
    if len(parts) < 2:
        return None, None
    algo_key = '-'.join(parts[-2:])
    regime = NOTE_REGIME.get(algo_key)
    if regime is None:
        return None, None
    hw_config = '-'.join(parts[:-2]) if len(parts) > 2 else 'baseline'
    return hw_config, regime


def plot_hardware_approach(df_sim, df_dynapse=None, hw_configs=None, include_baseline=True,
                           y_min=0.05, y_max=0.85, config_order=HW_APPROACH_ORDER,
                           config_labels=None, legend_loc='upper left'):
    """Grouped box plot showing how modeling hardware non-idealities approaches DYNAP-SE across
    Frozen, DFA, and Rec-LRA learning paradigms.

    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulation dataframe loaded from ``mnist1d_approaching_hardware.csv`` (with ``note`` and
        ``test accuracy (Max)`` / ``eval accuracy`` columns).
    df_dynapse : pd.DataFrame, optional
        DYNAP-SE dataframe (e.g. ``mnist1d_accuracy.csv``) processed via ``compute_best_val`` and
        appended as the final group on the right when provided.
    hw_configs : str or sequence of str, optional
        Hardware non-ideality names (as used in the CSV ``note`` prefix, e.g.
        ``['dpi_sat', 'w_clip', 'ei_asym', 'full_hw_quant']``) to plot. When ``None`` (default),
        all configurations in the CSV are plotted in ``config_order``.
    include_baseline : bool, default True
        Whether to prepend the software ``'baseline'`` column when ``hw_configs`` is specified.
    """
    from matplotlib.patches import Patch

    df = df_sim.copy()
    if 'test accuracy (Max)' not in df.columns:
        df['test accuracy (Max)'] = np.nan
    acc = pd.to_numeric(df['test accuracy (Max)'], errors='coerce')
    if 'eval accuracy' in df.columns:
        acc = acc.fillna(pd.to_numeric(df['eval accuracy'], errors='coerce'))
    if 'test accuracy (max)' in df.columns:
        acc = acc.fillna(pd.to_numeric(df['test accuracy (max)'], errors='coerce'))
    # Guard against locale-formatted values (e.g. 625.0 instead of 0.625).
    acc = acc.where(acc <= 1.0, np.where(acc > 100.0, acc / 1000.0, acc / 100.0))
    df['_accuracy'] = acc

    parsed = df['note'].apply(_parse_hw_approach_note)
    df['_hw_config'] = [p[0] for p in parsed]
    df['_regime'] = [p[1] for p in parsed]
    df = df.dropna(subset=['_hw_config', '_regime', '_accuracy'])

    available_configs = set(df['_hw_config'].unique())
    if hw_configs is not None:
        if isinstance(hw_configs, str):
            hw_configs = [hw_configs]
        requested = list(hw_configs)
        if include_baseline and 'baseline' in available_configs and 'baseline' not in requested:
            requested = ['baseline'] + requested
        groups = [c for c in requested if c in available_configs]
    else:
        groups = [c for c in config_order if c in available_configs]
        for c in df['_hw_config'].unique():
            if c not in groups:
                groups.append(c)

    best_dynapse = None
    if df_dynapse is not None and not df_dynapse.empty:
        best_dynapse = compute_best_val(df_dynapse, 'accuracy', 'higher')
        groups.append('dynapse')

    regimes = [r for r in REGIME_COLOR if r in set(df['_regime'].unique())]
    n_groups = len(groups)

    width_fraction = max(0.6, n_groups * 0.14)
    height_ratio = min(0.7, 0.38 * max(1.0, 12.0 / max(1, n_groups)))
    latexify(width_fraction=width_fraction, base_font_size=9, height_ratio=height_ratio)
    fig, ax = plt.subplots(constrained_layout=True)

    group_width = 0.78
    box_step = group_width / len(regimes)
    offsets = (np.arange(len(regimes)) - (len(regimes) - 1) / 2) * box_step
    x = np.arange(n_groups)

    for regime, offset in zip(regimes, offsets):
        sim_data, sim_pos = [], []
        dyn_data, dyn_pos = [], []
        for i, grp in enumerate(groups):
            if grp == 'dynapse':
                d = best_dynapse[best_dynapse['regime'] == regime]['accuracy'].dropna().values
                if len(d) > 0:
                    dyn_data.append(d)
                    dyn_pos.append(x[i] + offset)
            else:
                d = df[(df['_hw_config'] == grp) & (df['_regime'] == regime)]['_accuracy'].dropna().values
                if len(d) > 0:
                    sim_data.append(d)
                    sim_pos.append(x[i] + offset)
        if sim_data:
            _boxplot(ax, sim_data, sim_pos, box_step * 0.82, _lighten(REGIME_COLOR[regime], factor=0.35))
        if dyn_data:
            _boxplot(ax, dyn_data, dyn_pos, box_step * 0.82, REGIME_COLOR[regime])

    # Subtle vertical dividers separating Software baseline and Dynapse from intermediate configs.
    if 'baseline' in groups and groups.index('baseline') == 0 and n_groups > 1:
        ax.axvline(0.5, color='#888888', linestyle='--', linewidth=0.9, alpha=0.6)
    if 'dynapse' in groups and groups.index('dynapse') == n_groups - 1 and n_groups > 1:
        ax.axvline(n_groups - 1.5, color='#888888', linestyle='--', linewidth=0.9, alpha=0.6)

    labels_lookup = {**HW_APPROACH_LABEL, **(config_labels or {})}
    tick_labels = [labels_lookup.get(g, g.replace('_', ' ')) for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Best validation accuracy')
    ax.tick_params(axis='both', labelsize=8)
    _sparse_grid(ax, axis='y')

    legend_handles = [Patch(facecolor=REGIME_COLOR[r], label=REGIME_LABEL[r]) for r in regimes]
    ax.legend(handles=legend_handles, loc=legend_loc, ncol=len(regimes))

    return fig

