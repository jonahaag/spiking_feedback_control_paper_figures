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

COLOR_TARGET = '#CC6677'      # rose (3f, reused for predicted-class 2f)
COLOR_NONTARGET = '#44AA99'   # teal (3f, reused for predicted-class 2f)
COLOR_HIDDEN = '#332288'      # indigo -- hidden features (2d/4d) and input-to-hidden weights (2e)
COLOR_HIDDEN_OUT = '#88CCEE'  # cyan -- hidden-to-output weights (2e)
COLOR_FEEDBACK = '#882255'    # wine -- feedback weights (2e)

CMAP_HIDDEN = LinearSegmentedColormap.from_list('hidden', [(1, 1, 1, 1), to_rgba(COLOR_HIDDEN)], N=100)  # (2d/4d)

REGIME_COLOR = {'full': COLOR_FULL, 'frozen': COLOR_FROZEN}
REGIME_LABEL = {'full': 'Full', 'frozen': 'Frozen'}
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


def plot_accuracy_over_steps(agg, ax=None, reference_lines=None, chance_level=None):
    """Accuracy curves (center +/- band, typically from ``aggregate_mean_std``): color = regime.
    Training is a solid line with a shaded +/- band; validation is scattered markers only (mean,
    no error bar/band).

    ``chance_level``, if given, prepends a step-0 point (untrained network) to both curves.
    """
    if ax is None:
        latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
        _, ax = plt.subplots(constrained_layout=True)

    for regime, color in REGIME_COLOR.items():
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


# --- grouped bar charts (4b/4c) -------------------------------------------------------------------

def compute_best_val(df, value_col, better):
    """Best validation value (max if better='higher', min if better='lower') per run."""
    val = df[df['split'] == 'val']
    agg = 'max' if better == 'higher' else 'min'
    return val.groupby(['regime', 'seed', 'chip'], as_index=False)[value_col].agg(agg)


def plot_grouped_bars_by_chip(df):
    """Best validation accuracy, Full vs. Frozen, grouped by chip.

    Bar height = mean across seeds within that chip, error bar = std. Shows whether results are
    consistent across chips.
    """
    best = compute_best_val(df, 'accuracy', 'higher')

    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)
    width = 0.38

    values = sorted(best['chip'].unique())
    x = np.arange(len(values))
    stats = best.groupby(['regime', 'chip'])['accuracy'].agg(['mean', 'std']).reindex(
        pd.MultiIndex.from_product([REGIME_COLOR, values], names=['regime', 'chip']))

    for regime, color in REGIME_COLOR.items():
        s = stats.loc[regime]
        offset = (-width / 2 if regime == 'frozen' else width / 2)
        ax.bar(x + offset, s['mean'], width, yerr=s['std'].fillna(0), capsize=3,
              color=color, label=REGIME_LABEL[regime], error_kw={'elinewidth': 1})

    ax.set_xticks(x)
    ax.set_xticklabels([f'Chip {v}' for v in values])
    ax.set_ylabel('Best validation accuracy')
    span = best['accuracy'].max() - best['accuracy'].min()
    ax.set_ylim(max(0, best['accuracy'].min() - 0.3 * span), best['accuracy'].max() + 0.3 * span)
    _sparse_grid(ax, axis='y')
    ax.legend()

    return fig


# --- weight distributions (2e/3d) -----------------------------------------------------------------

# Reused across datasets: color/title assigned by subplot position (earliest -> latest weight
# matrix), not by semantic weight name, so different weight-graph shapes (e.g. yin-yang's single
# hidden layer + feedback vs. mnist1d's two hidden layers) can share the same visual identity.
YINYANG_WEIGHT_COLOR = {'in_hidden': COLOR_HIDDEN, 'hidden_out': COLOR_HIDDEN_OUT, 'feedback': COLOR_FEEDBACK}
YINYANG_WEIGHT_TITLE = {'in_hidden': 'Input → hidden', 'hidden_out': 'Hidden → output', 'feedback': 'Feedback'}

MNIST1D_WEIGHT_COLOR = {'in_hid1': COLOR_HIDDEN, 'hid1_hid2': COLOR_HIDDEN_OUT, 'hid2_out': COLOR_FEEDBACK}
MNIST1D_WEIGHT_TITLE = {'in_hid1': 'Input → hidden 1', 'hid1_hid2': 'Hidden 1 → hidden 2',
                        'hid2_out': 'Hidden 2 → output'}

PHASE_LABEL = {'init': 'Initialization', 'end': 'End of training'}
PHASE_FILL_ALPHA = {'init': 0.25, 'end': 0.45}
PHASE_LINESTYLE = {'init': ':', 'end': '-'}


def plot_weight_distributions(df, colors, titles, pad_fraction=0.15, bw_adjust=0.6):
    """1x3 grid: KDE-smoothed weight distribution at initialization vs. end of training, one
    panel per weight type (``colors``/``titles`` keys, in order), each in its own color. The area
    where the two distributions overlap is hatched to make it easy to spot.

    ``bw_adjust`` scales the KDE bandwidth relative to scipy's default (Scott's rule); less than
    1 makes the curve track the data more closely (less smoothing).
    """
    latexify(width_fraction=1.0, base_font_size=9, height_ratio=0.4)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    for i, (ax, weight_type) in enumerate(zip(axes, colors)):
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

def plot_sim_vs_dynapse_bars(df_dynapse, df_sim, y_min=0.0, y_max=1.0, group_order=None):
    """Bar plot comparing Simulation vs. Dynapse for various methods.

    Expects df_sim to contain 'note' and 'test accuracy (Max)'
    Expects df_dynapse to contain 'regime' and 'accuracy' (processed via compute_best_val)
    """
    if group_order is None:
        group_order = ['locally-learnt', 'frozen-hidden', 'fixed-feedback', 'surrogate gradient', 'backprop']
        
    # Process Dynapse
    best_dynapse = compute_best_val(df_dynapse, 'accuracy', 'higher')
    regime_map = {'full': 'locally-learnt', 'frozen': 'frozen-hidden'}
    if 'note' not in best_dynapse.columns:
        best_dynapse['note'] = best_dynapse['regime'].map(regime_map).fillna(best_dynapse['regime'])
    dynapse_stats = best_dynapse.groupby('note')['accuracy'].agg(['mean', 'std']).reindex(group_order)
    
    # Process Simulation
    sim_stats = df_sim.groupby('note')['test accuracy (Max)'].agg(['mean', 'std']).reindex(group_order)
    
    # Style matching plot_grouped_bars_by_chip exactly
    latexify(width_fraction=0.6, base_font_size=9, height_ratio=0.7)
    fig, ax = plt.subplots(constrained_layout=True)
    width = 0.38
    
    # Method color mapping using Paul Tol palette defined at top of file
    method_color_map = {
        'locally-learnt': COLOR_FULL,
        'frozen-hidden': COLOR_FROZEN,
        'fixed-feedback': COLOR_NONTARGET,
        'surrogate gradient': COLOR_HIDDEN_OUT,
        'backprop': COLOR_REFERENCE,
    }
    
    def _lighten(color_hex, factor=0.45):
        from matplotlib.colors import to_rgb, to_hex
        rgb = np.array(to_rgb(color_hex))
        return to_hex((1 - factor) * rgb + factor * 1.0)
    
    x = np.arange(len(group_order))
    
    for i in range(len(x)):
        note = group_order[i]
        d_mean = dynapse_stats.iloc[i]['mean']
        d_std = dynapse_stats.iloc[i]['std']
        s_mean = sim_stats.iloc[i]['mean']
        s_std = sim_stats.iloc[i]['std']
        
        c_dyn = method_color_map.get(note, COLOR_FULL)
        c_sim = _lighten(c_dyn)
        
        # If dynapse data is present, plot side-by-side
        if pd.notna(d_mean) and d_mean > 0:
            ax.bar(x[i] - width/2, s_mean, width, yerr=0 if pd.isna(s_std) else s_std, 
                   capsize=3, color=c_sim, error_kw={'elinewidth': 1})
            ax.bar(x[i] + width/2, d_mean, width, yerr=0 if pd.isna(d_std) else d_std, 
                   capsize=3, color=c_dyn, error_kw={'elinewidth': 1})
        else:
            # Center the bar if dynapse is missing
            ax.bar(x[i], s_mean, width, yerr=0 if pd.isna(s_std) else s_std, 
                   capsize=3, color=c_sim, error_kw={'elinewidth': 1})

    ax.set_xticks(x)
    label_map = {
        'locally-learnt': f'Full',
        'frozen-hidden': f'Frozen',
        'fixed-feedback': f'Random\nFeedback',
        'surrogate gradient': f'Surrogate\nGradient',
        'backprop': f'ANN\n(SGD)'
    }
    tick_labels = [label_map.get(note, str(note).replace('-', ' ').title()) for note in group_order]
    # ax.set_xticklabels(tick_labels, rotation=15, ha='center', rotation_mode='anchor')
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')
    ax.set_ylabel('Best validation accuracy')
    ax.tick_params(axis='both', labelsize=8)
    # ax.tick_params(axis='x', pad=6)
    
    ax.set_ylim(y_min, y_max)
    _sparse_grid(ax, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_lighten('#555555'), label='Simulation'),
        Patch(facecolor='#555555', label='Dynapse')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    return fig
