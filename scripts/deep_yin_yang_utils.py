import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# plt.style.use('default')
import os
import matplotlib
from matplotlib.cm import get_cmap

def latexify(width_in_points=5.91, width_fraction=1.0, height_ratio=None, base_font_size=10):
    """
    Set up matplotlib's RC params to match LaTeX document appearance.

    Parameters
    ----------
    width_in_points : float
        Document text width in inches.
    width_fraction : float
        Fraction of the text width the figure should occupy.
    height_ratio : float or None
        Height/width ratio. Uses golden ratio if None.
    base_font_size : int
        Font size to match LaTeX document (e.g., 10 for 11pt LaTeX class).
    """
    fig_width = width_in_points * width_fraction
    if height_ratio is None:
        golden_ratio = (5**0.5 - 1) / 2
        fig_height = fig_width # * golden_ratio
    else:
        fig_height = fig_width * height_ratio

    plt.rcParams.update({
        "figure.figsize": (fig_width, fig_height),
        # "font.family": "serif",
        # "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times"],
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

def calculate_rolling_mean(data : np.ndarray, rolling_window : int = 1):
    """
    Calculate the rolling mean of a 1D array. Could also do this with np.convolve, but their indexing is confusing.
    """
    rolling_mean = np.zeros_like(data)
    rolling_std = np.zeros_like(data)
    # Not very efficient, but does the trick.
    for i in range(len(data)):
        i1 = max(0, i - rolling_window)
        rolling_mean[i] = data[i1:i+1].mean()
        rolling_std[i] = data[i1:i+1].std()
    return rolling_mean, rolling_std

def plot_loss(trainloss : np.ndarray, valloss : np.ndarray, testloss : np.ndarray,
              trainloss_std: np.ndarray, valloss_std: np.ndarray, testloss_std : np.ndarray,
              rolling_window : int = 1, title : str = None, ylabel : str = None, savepath : str = None,
              scale : str = None, legendFLAG : bool = True, ylim : tuple = (None, None),
              val_frequency : int = 1):
    """
    Plot the training and validation loss over batches.
    Args:
        trainloss (np.ndarray): An array of shape (n_batches, batch_size) containing the training loss during training.
        valloss (np.ndarray): An array of shape (n_batches, batch_size) containing the validation loss during training.
        testloss (np.ndarray): An array of shape (n_batches, batch_size) containing the test loss during training.
        rolling_window (int, optional): Rolling window length. Defaults to 1.
        title (str, optional): Figure title. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        scale (str, optional): Scale of the y-axis. Defaults to None, in which case the scale is linear. If 'log', the scale is logarithmic.
    """

    # Average for each batch
    if trainloss.ndim == 2:
        trainloss = trainloss.mean(axis=1)
    if valloss.ndim == 2:
        valloss_mean = valloss.mean(axis=1)
    # Average across all test samples
    testloss_mean = testloss.mean()

    batches = np.arange(1,trainloss.shape[0]+1)
    batches_val = np.arange(0,valloss.shape[0]+0)
    trainloss_mean, _ = calculate_rolling_mean(trainloss, rolling_window)

    latexify(width_fraction=0.33, base_font_size=9, height_ratio=1.)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1.)
    linewidth = 1.5
    batch_size = 20
    # ax.fill_between(batches_val * batch_size * val_frequency, valloss_mean-valloss_std, valloss_mean+valloss_std, alpha=0.2, color='#5E3493')
    ax.fill_between(batches * batch_size, trainloss_mean-trainloss_std, trainloss_mean+trainloss_std, alpha=0.2, color='#05538D')
    ax.fill_between(batches * batch_size, testloss_mean-testloss_std, testloss_mean+testloss_std, alpha=0.1, color='black')
    ax.plot(batches * batch_size, np.ones_like(batches)*testloss_mean, '--', c='black', label='Test', linewidth=1., alpha=0.8)
    ax.plot(batches * batch_size, trainloss_mean, label='Train', linewidth=linewidth, color='#05538D')
    ax.plot(batches_val * batch_size * val_frequency, valloss_mean, label='Validation', linewidth=linewidth, color='#5E3493')
    ax.set_xlabel('# examples')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, batches[-1] * batch_size + 1, batch_size * 500))
    ax.set_xlim(-1, batches[-1] * batch_size * 1.03)
    ax.set_ylim(ylim[0], ylim[1])
    # ax.set_xscale('log')
    ax.grid(True)
    # ax.set_yscale('log')
    if legendFLAG:
        ax.legend()
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_weights_in_hidden(weights : np.ndarray, epoch_step : int = 50, savepath : str = None, cmap : str = 'Accent',
                            vmin = -6, vmax = 6):
    """
    Plot a histogram of the weights of the network after every {epoch_step} batches in a new subfigure, split by output.
    Args:
        weights (np.ndarray): An array of shape (n_batches, batch_size, 4, n_hidden) containing the integer weights during training.
        epoch_step (int, optional): "Stepsize" specifying the number of batches between two plots. Defaults to 50.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
    """
    n_batches = weights.shape[0]
    batch_size = weights.shape[1]
    weights = weights.mean(axis=1)

    epoch_ids = np.arange(0,n_batches-1,epoch_step).astype(int)
    latexify(width_fraction=1.5, base_font_size=7, height_ratio=0.618)
    fig, ax = plt.subplots(3,5,constrained_layout=True)
    for i in range(15):
        j,k = divmod(i,5)
        ax[j,k].imshow(weights[epoch_ids,:,i].T, aspect='auto',
                     cmap='RdBu', interpolation='none',
                     vmin=vmin, vmax=vmax)
        ax[j,k].set_title(f'Hidden neuron {i}', fontsize=10)
        ax[j,k].set_xlabel('# examples') if j == 2 else None
        ax[j,k].set_xticks(np.arange(int(500/epoch_step),len(epoch_ids),int(500/epoch_step))) if j == 2 else ax[j,k].set_xticks([])
        ax[j,k].set_xticklabels([str(e*batch_size) for e in np.arange(500, n_batches, 500)], rotation=30, fontsize=8) if j == 2 else ax[j,k].set_xticklabels([])
        ax[j,k].set_yticks([0,1,2,3])
        ax[j,k].set_yticklabels(['x', 'y', '1-x', '1-y']) if j == 0 else None
    # ax[i].set_ylabel('Hidden neuron')
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    plt.close(fig)

def plot_weights_hidden_out(weights : np.ndarray, epoch_step : int = 50, savepath : str = None, cmap : str = 'Accent',
                            vmin = -6, vmax = 6):
    """
    Plot a histogram of the weights of the network after every {epoch_step} batches in a new subfigure, split by output.
    Args:
        weights (np.ndarray): An array of shape (n_batches, batch_size, n_hidden, 3) containing the integer weights during training.
        epoch_step (int, optional): "Stepsize" specifying the number of batches between two plots. Defaults to 50.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
    """
    batch_sizees = weights.shape[0]
    batch_size = weights.shape[1]
    weights = weights.mean(axis=1)

    epoch_ids = np.arange(0,batch_sizees-1,epoch_step).astype(int)
    latexify(width_fraction=1., base_font_size=9, height_ratio=0.3)
    fig, ax = plt.subplots(1,4,width_ratios=[1,1,1,0.05], gridspec_kw={'wspace':0.1})
    for i in range(3):
        # ax[i].set_box_aspect(0.618)
        ax[i].imshow(weights[epoch_ids,:,i].T, aspect='auto',
                     cmap='RdBu', interpolation='none',
                     vmin=vmin, vmax=vmax)
        ax[i].set_xticks(np.arange(int(1000/epoch_step),len(epoch_ids),int(1000/epoch_step)))
        ax[i].set_xticklabels([str(e*batch_size) for e in np.arange(1000, batch_sizees, 1000)])
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        ax[i].set_title(['Yin', 'Yang', 'Dot'][i])
    ax[0].set_ylabel('Hidden neuron')
    ax[1].set_xlabel('# examples')
    ax[3].set_axis_off()
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                                              cmap='RdBu'),
                 orientation='vertical',
                 label='Weight',
                 fraction=1.,
                 ax=ax[-1])
    # fig.colorbar(im, ax=ax[-1])
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_weights_in_hidden_lines(weights, savepath):
    n_batches = weights.shape[0]
    batch_size = weights.shape[1]
    weights = weights.mean(axis=1)
    batches = np.arange(1,weights.shape[0]+1)

    latexify(width_fraction=1.5, base_font_size=9, height_ratio=0.618)
    fig, ax = plt.subplots(3,5,constrained_layout=True)
    # ax.set_box_aspect(1.)
    linewidth = 1.
    cmap = get_cmap('Dark2', 4)
    colors = [cmap(i) for i in range(4)]
    vmax = np.abs(weights).max()
    for i in range(15):
        j, k = divmod(i, 5)
        ax[j,k].set_prop_cycle('color',colors)
        ax[j,k].plot(batches, weights[:, :, i], linewidth=linewidth)
        ax[j,k].set_title(f'Hidden neuron {i}', fontsize=10)
        ax[j,k].set_xlabel('# examples') if j == 2 else None
        ax[j,k].set_xticks(np.arange(int(500),len(batches),int(500)))
        ax[j,k].set_xticklabels([str(e*batch_size) for e in np.arange(500, n_batches, 500)], rotation=30, fontsize=8) if j == 2 else ax[j,k].set_xticklabels([])
        ax[j,k].set_ylim(-vmax*1.1,vmax*1.1)
        ax[j,k].grid(True)

    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    plt.close(fig)

def plot_weights_hidden_out_lines(weights, savepath):
    n_batches = weights.shape[0]
    batch_size = weights.shape[1]
    weights = weights.mean(axis=1)
    batches = np.arange(1,weights.shape[0]+1)

    latexify(width_fraction=1., base_font_size=9, height_ratio=0.3)
    fig, ax = plt.subplots(1,3,constrained_layout=True)
    linewidth = 1.
    cmap = get_cmap('Dark2', 15)
    colors = [cmap(i) for i in range(15)]
    vmax = np.abs(weights).max()
    for i in range(3):
        ax[i].set_prop_cycle('color', colors)
        ax[i].plot(batches, weights[:, :, i], linewidth=linewidth)
        ax[i].set_title(['Yin', 'Yang', 'Dot'][i])
        ax[i].set_xlabel('# examples') if i == 2 else None
        ax[i].set_xticks(np.arange(int(500),len(batches),int(500)))
        ax[i].set_xticklabels([str(e*batch_size) for e in np.arange(500, n_batches, 500)], rotation=30, fontsize=8)
        ax[i].set_ylim(-1.1*vmax,1.1*vmax)
        ax[i].grid(True)
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    plt.close(fig)

def plot_weight_changes(weights, savepath):
    n_batches = weights.shape[0]
    batch_size = weights.shape[1]
    weights = weights[:,-1,:,:]
    batches = np.arange(1,weights.shape[0])

    weights = np.rint(weights)
    weight_changes = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    weight_changes = calculate_rolling_mean(weight_changes, rolling_window=50)[0]

    latexify(width_fraction=1., base_font_size=9, height_ratio=0.3)
    fig, ax = plt.subplots(1,3,constrained_layout=True)
    linewidth = 2.
    cmap = get_cmap('Dark2', 15)
    colors = [cmap(i) for i in range(15)]
    for i in range(3):
        ax[i].set_prop_cycle('color', colors)
        ax[i].plot(batches, weight_changes[:,i], linewidth=linewidth)
        # ax[i].imshow(weight_changes[:, :, i].T, aspect='auto', cmap='RdBu_r',
        #              interpolation='none', vmin=-5, vmax=5)
        ax[i].set_title(['Yin', 'Yang', 'Dot'][i])
        ax[i].set_xlabel('# examples')
        ax[i].set_xticks(np.arange(int(500),len(batches),int(500)))
        ax[i].set_xticklabels([str(e*batch_size) for e in np.arange(500, n_batches, 500)], rotation=30, fontsize=8)
        # ax[i].set_ylim(-5,5)
        ax[i].grid(True)
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    plt.close(fig)

def plot_controls(cp : np.ndarray, cn : np.ndarray, labels : np.ndarray, out_idx : int = None,
                  rolling_window : int = 1, title : str = None, savepath : str = None, cmap : str = 'Accent'):
    """
    Plot the difference control signals of the network over batches, split by examples from different classes.
    Args:
        cp (np.ndarray): An array of shape (n_batches, batch_size, n_pop, n_out) containing the positive control activities during training.
        cn (np.ndarray): An array of shape (n_batches, batch_size, n_pop, n_out) containing the negative control activities during training.
        labels (np.ndarray): An array of shape (n_batches*batch_size) containing the labels for each sample.
        out_idx (int, optional): Only plot weights of connections pointing to a specific output. Defaults to 0.
        rolling_window (int, optional): Rolling window length. Defaults to 1.
        title (str, optional): Figure title. Defaults to None.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
    """
    cmap = ListedColormap(["#75157C", "#6F93E6", "#CC76D1"])
    lcmap = plt.cm.get_cmap(cmap, 3)
    n_batches = cp.shape[0]
    batch_size = cp.shape[1]

    batches = np.arange(1,n_batches+1)
    latexify(width_fraction=0.99, base_font_size=9, height_ratio=1/3)
    fig, ax = plt.subplots(1,3,sharey=True,constrained_layout=True)
    linewidth = 2.
    # Pick all activity only of the controls related to output out_idx
    for out_idx in range(3):
        ax[out_idx].set_box_aspect(0.618)
        if out_idx is not None:
            cp_samples = cp[:,:,:,out_idx]
            cn_samples = cn[:,:,:,out_idx]
        else:
            cp_samples = cp
            cn_samples = cn
        
        cp_mean = np.zeros(n_batches)
        cp_std = np.zeros(n_batches)
        cn_mean = np.zeros(n_batches)
        cn_std = np.zeros(n_batches)
        cdiff_mean = np.zeros(n_batches)
        cdiff_std = np.zeros(n_batches)

        # ax.plot(batches, np.zeros_like(batches), '--', c='gray')
        c_diff_mean_sum = np.zeros(n_batches)
        for c in range(3): # Examples from class c
            for i in range(n_batches):
                # Pick all labels for the current epoch
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                # Pick samples from class c in the current epoch
                class_idx = np.where(batch_labels == c)[0]
                if len(class_idx)==0:
                    cp_mean[i] = cp_mean[i-1]
                    cp_std[i] = cp_std[i-1]
                    cn_mean[i] = cn_mean[i-1]
                    cn_std[i] = cn_std[i-1]
                    cdiff_mean[i] = cdiff_mean[i-1]
                    cdiff_std[i] = cdiff_std[i-1]
                else: 
                    cp_samples_class = cp_samples[i,class_idx,:]
                    cn_samples_class = cn_samples[i,class_idx,:]
                    # Calculate the mean and std (across the population) of the positive/negative control signals and their difference
                    cp_mean[i] = cp_samples_class.mean()
                    cp_std[i] = cp_samples_class.std()
                    cn_mean[i] = cn_samples_class.mean()
                    cn_std[i] = cn_samples_class.std()
                    cdiff_mean[i] = cp_samples_class.mean() - cn_samples_class.mean()
                    cdiff_std[i] = (cp_samples_class - cn_samples_class).std()
            cdiff_mean, _ = calculate_rolling_mean(cdiff_mean, rolling_window=rolling_window)
            c_diff_mean_sum += cdiff_mean
            label = ['Yin', 'Yang', 'Dot'][c]
            ax[out_idx].plot(batch_size*batches, cdiff_mean, label=label, c=lcmap(c), linewidth=linewidth)
            # ax[out_idx].fill_between(batch_size*batches, cdiff_mean-cdiff_std, cdiff_mean+cdiff_std, alpha=0.3, color=lcmap(c))
        ax[out_idx].plot(batch_size*batches, c_diff_mean_sum, c='black', alpha=0.5, linewidth=linewidth, label='Sum')
        ax[out_idx].set_xticks(np.arange(0, batch_size * batches[-1] + 1, batch_size * 500))
        # ax[out_idx].set_xlim(-500, batch_size * batches[-1] + 500)
        ax[out_idx].set_xlim(1*batch_size, batch_size * batches[-1] + 500)
        ax[out_idx].set_xscale('log')
        ax[out_idx].grid(True)
        ax[out_idx].set_xlabel('# examples')
        ax[out_idx].set_title(f"Output {['Yin', 'Yang', 'Dot'][out_idx]}")
    ax[0].set_ylabel('Feedback [Hz]')
    ax[2].legend(ncol=2, title="Example from class")
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_control_summary(cp : np.ndarray, cn : np.ndarray, labels : np.ndarray, out_idx : int = None,
                         rolling_window : int = 1, savepath : str = None, cmap : str = 'Accent'):
    """
    Plot the difference control signals of the network over batches, split by examples from different classes.
    Args:
        cp (np.ndarray): An array of shape (n_batches, batch_size, n_pop, n_out) containing the positive control activities during training.
        cn (np.ndarray): An array of shape (n_batches, batch_size, n_pop, n_out) containing the negative control activities during training.
        labels (np.ndarray): An array of shape (n_batches*batch_size) containing the labels for each sample.
        out_idx (int, optional): Only plot weights of connections pointing to a specific output. Defaults to 0.
        rolling_window (int, optional): Rolling window length. Defaults to 1.
        title (str, optional): Figure title. Defaults to None.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
    """
    lcmap = plt.cm.get_cmap(cmap, 3)
    n_batches = cp.shape[0]
    batch_size = cp.shape[1]

    batches = np.arange(1,n_batches+1)
    latexify(width_fraction=0.33, base_font_size=9, height_ratio=1.)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1.)
    linewidth = 2.
    # Pick all activity only of the controls related to output out_idx
    cdiff_mean = np.zeros(n_batches)
    cdiff_std = np.zeros(n_batches)
    c_diff_mean_sum = np.zeros((n_batches, 3))

    # cp = cp.mean(axis=2).mean(axis=1).mean(axis=1)
    # cn = cn.mean(axis=2).mean(axis=1).mean(axis=1)
    # ax.plot(batches * batch_size, np.abs(calculate_rolling_mean(cp-cn,rolling_window)[0]), linewidth=linewidth, c='black')

    for out_idx in range(3):
        if out_idx is not None:
            cp_samples = cp[:,:,:,out_idx]
            cn_samples = cn[:,:,:,out_idx]
        else:
            cp_samples = cp
            cn_samples = cn

        for c in range(3): # Examples from class c
            for i in range(n_batches):
                # Pick all labels for the current epoch
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                # Pick samples from class c in the current epoch
                class_idx = np.where(batch_labels == c)[0]
                if len(class_idx)==0:
                    cdiff_mean[i] = cdiff_mean[i-1]
                    cdiff_std[i] = cdiff_std[i-1]
                else: 
                    cp_samples_class = cp_samples[i,class_idx,:]
                    cn_samples_class = cn_samples[i,class_idx,:]
                    cdiff_mean[i] = cp_samples_class.mean() - cn_samples_class.mean()
                    cdiff_std[i] = (cp_samples_class - cn_samples_class).std()
            cdiff_mean, _ = calculate_rolling_mean(cdiff_mean, rolling_window=rolling_window)
            c_diff_mean_sum[:,out_idx] += cdiff_mean
    
    mu = np.abs(c_diff_mean_sum.mean(axis=1))
    sigma = c_diff_mean_sum.std(axis=1)
    ax.plot(batches * batch_size, mu, linewidth=linewidth, c='black')
    ax.fill_between(batches * batch_size, mu-sigma, mu+sigma, alpha=0.3, color='black')
    ax.set_xscale('log')
    # ax.set_xticks(np.arange(0, batch_size * batches[-1] + 1, batch_size * 500))
    ax.set_xlim(1*batch_size, batch_size * batches[-1] + 500)
    ax.set_ylim(None, 23)
    ax.set_yticks(np.arange(0, 25, 5))
    ax.grid(True)
    ax.set_xlabel('# examples')
    ax.set_ylabel('Feedback [Hz]')
    # ax.set_yscale('log')
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_outputs(outputs : np.ndarray, labels : np.ndarray, target_frmin : int, target_frmax : int,
                 rolling_window : int = 1, title : str = None, savepath : str = None, cmap : str = 'Accent'):
    """
    Plot the activity of the single output over batches, split by examples from different classes.
    Args:
        outputs (np.ndarray): An array of shape (n_batches, batch_size, n_pop, n_out) containing the output activities during training.
        labels (np.ndarray): An array of shape (n_batches*batch_size) containing the labels for each sample.
        target_frmin (int): Minimum target firing rate.
        target_frmax (int): Maximum target firing rate.
        out_idx (int): Only plot weights of connections pointing to a specific output. Defaults to 0.
        rolling window (int, optional): Rolling window length. Defaults to 1.
        title (str, optional): Figure title. Defaults to None.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
    """
    cmap = ListedColormap(["#75157C", "#6F93E6", "#CC76D1"])
    lcmap = plt.cm.get_cmap(cmap, 3)
    n_batches = outputs.shape[0]
    batch_size = outputs.shape[1]

    latexify(width_fraction=0.99, base_font_size=9, height_ratio=1/3)
    fig, ax = plt.subplots(1,3,sharey=True,constrained_layout=True)
    batches = np.arange(1,n_batches+1)
    linewidth = 2.
    # Pick all activity only of the controls related to output out_idx
    for out_idx in range(3):
        ax[out_idx].set_box_aspect(0.618)
        output_samples = outputs[:,:,:,out_idx]
        
        out_mean = np.zeros(n_batches)
        out_std = np.zeros(n_batches)

        for c in range(3): # Examples from class c
            for i in range(n_batches):
                # Pick all labels for the current epoch
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                # Pick samples from class c in the current epoch
                class_idx = np.where(batch_labels == c)[0]
                if len(class_idx)==0:
                    out_mean[i] = out_mean[i-1]
                    out_std[i] = out_std[i-1]
                else:
                    # Calculate mean and std (across the population) of the output activity
                    out_mean[i] = output_samples[i,class_idx,:].mean()
                    out_std[i] = output_samples[i,class_idx,:].std()
            label = ['Yin', 'Yang', 'Dot'][out_idx]
            out_mean, _ = calculate_rolling_mean(out_mean, rolling_window=rolling_window)
            out_std, _ = calculate_rolling_mean(out_std, rolling_window=rolling_window)
            ax[c].plot(20 * batches, out_mean, label=label, c=lcmap(out_idx), linewidth=linewidth)
            # ax[c].fill_between(20 * batches, out_mean-out_std, out_mean+out_std, alpha=0.3, color=lcmap(out_idx))
        ax[out_idx].plot(20 * batches, target_frmin*np.ones_like(batches), '--', c='black', label='Target')
        ax[out_idx].plot(20 * batches, target_frmax*np.ones_like(batches), '--', c='black')
        ax[out_idx].set_xticks(np.arange(0, 20 * batches[-1] + 1, 20 * 500))
        ax[out_idx].set_xlim(1*20, 20 * batches[-1] + 500)
        ax[out_idx].set_xscale('log')
        ax[out_idx].grid(True)
        ax[out_idx].set_xlabel('# examples')
        ax[out_idx].set_title(f"Class {['Yin', 'Yang', 'Dot'][out_idx]}")
    
    ax[0].set_ylabel('Output [Hz]')
    ax[2].legend(ncol=2, title="Output", loc='upper right')
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_predictions(inputs : np.ndarray, predictions : np.ndarray, title : str = None,
                     cmap : str = 'Accent', input_frmin : int = 10, input_frmax : int = 100,
                     savepath : str = None, vmin : int = 0, vmax : int = 2):
    """
    Plot the predictions of the network over the input space.
    Args:
        inputs (np.ndarray): An array of shape (n_samples, x) where x is at least 2, containing the inputs to the network.
        predictions (np.ndarray): An array of length n_samples containing the predicted labels for each sample.
        title (str, optional): Figure title. Defaults to None.
        cmap (str, optional): Colormap to use. Defaults to 'Accent'.
        savepath (str, optional): Path and filename to store the figure. Defaults to None, in which case the figure is not saved.
        vmin (int, optional): Minimum value for color mapping. Defaults to 0.
        vmax (int, optional): Maximum value for color mapping. Defaults to 2.
    """
    cmap = ListedColormap(["#75157C", "#6F93E6", "#CC76D1"])
    latexify(width_fraction=0.33, base_font_size=9, height_ratio=1.)
    fig, ax =  plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1.)
    ax.hlines(y=[input_frmin,input_frmax], xmin=input_frmin-10, xmax=input_frmax+10, color='black', linewidth=.5, alpha=1.)
    ax.vlines(x=[input_frmin,input_frmax], ymin=input_frmin-10, ymax=input_frmax+10, color='black', linewidth=.5, alpha=1.)
    ax.scatter(inputs[:,0], inputs[:,1], c=predictions, cmap=cmap, alpha=1., s=2, vmin=vmin, vmax=vmax)
    ax.set_xlim(input_frmin-6, input_frmax+6)
    ax.set_ylim(input_frmin-6, input_frmax+6)
    ax.set_title(title)
    ax.set_xlabel('x [Hz]')
    ax.set_ylabel('y [Hz]')
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_hidden_activity(inputs : np.ndarray, hidden_activity : np.ndarray, savepath : str = None):
    """
    ...
    Args:
        inputs (np.ndarray): _description_
        hidden_activity (np.ndarray): _description_
        savepath (str, optional): _description_. Defaults to None.
    """
    n_hidden = hidden_activity.shape[3]
    hidden_activity = hidden_activity.mean(axis=2)
    hidden_activity = hidden_activity.reshape(-1,n_hidden)
    vmin = hidden_activity[:,:-1].min()
    vmax = hidden_activity[:,:-1].max()
    denom = vmax - vmin if vmax > vmin else 1.

    latexify(width_fraction=1., base_font_size=9, height_ratio=3/5)
    fig, ax = plt.subplots(3, 5)
    for i in range(n_hidden):
        # vmin = hidden_activity[:,i].min()
        # vmax = hidden_activity[:,i].max()
        j,k = divmod(i,5)
        alpha = np.clip((hidden_activity[:,i]-vmin)/denom, 0., 1.)
        # ax[j,k].scatter(inputs[:,0], inputs[:,1], c='Black',
        #                 alpha=alpha, s=20.)
        ax[j,k].set_axis_off()
        # ax[j,k].scatter(inputs[:,0], inputs[:,1], c=hidden_activity[:,i],
        #                        cmap='Greys', alpha=0.5, s=10, vmin=vmin, vmax=vmax)

        lcmap = LinearSegmentedColormap.from_list("Custom", [(1,1,1,1), (0,0,0,1)], N=100)
        ax[j,k].scatter(inputs[:,0], inputs[:,1], c=hidden_activity[:,i],
                        alpha=1., s=20, cmap=lcmap, vmin=vmin, vmax=vmax)
    # for j in range(round(n_hidden/5)):
    #     for k in range(6):     
    #         ax[j,k].set_axis_off()
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    #                                           cmap='Black'),
    #              ax=ax[:,-1],
    #              orientation='vertical',
    #              fraction=1.8)
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    plt.close(fig)

def plot_output_feature_maps(inputs : np.ndarray, output_activity : np.ndarray, 
                            input_frmin : int = 10, input_frmax : int = 100,
                            cmap =["#75157C", "#6F93E6", "#CC76D1"], savepath : str = None):
    output_activity = output_activity.mean(axis=2).reshape(-1,3)
    latexify(width_fraction=0.99, base_font_size=9, height_ratio=1/3)
    fig, ax = plt.subplots(1,3,constrained_layout=True,sharey=True,width_ratios=[1,1,1],gridspec_kw={'wspace':0.15}) 
    vmin = output_activity.min()
    vmax = output_activity.max()
    for out_idx in range(3):
        ax[out_idx].set_box_aspect(1.)
        ax[out_idx].hlines(y=[input_frmin,input_frmax], xmin=input_frmin-10, xmax=input_frmax+10, color='black', linewidth=.5, alpha=1.)
        ax[out_idx].vlines(x=[input_frmin,input_frmax], ymin=input_frmin-10, ymax=input_frmax+10, color='black', linewidth=.5, alpha=1.)
        ax[out_idx].set_xlim(input_frmin-6, input_frmax+6)
        ax[out_idx].set_ylim(input_frmin-6, input_frmax+6)
        if out_idx == 1:
            ax[out_idx].set_xlabel('x [Hz]')
        if out_idx == 0:
            ax[out_idx].set_ylabel('y [Hz]')
        # if out_idx != 0:
        #     ax[out_idx].set_yticks([])
        # ax[out_idx].scatter(inputs[:,0], inputs[:,1], c=cmap[out_idx],
        #                     alpha=(output_activity[:,out_idx]-vmin)/(vmax-vmin), s=10.)

        lcmap = LinearSegmentedColormap.from_list("Custom", [(1,1,1,1), cmap[out_idx]], N=100)
        ax[out_idx].scatter(inputs[:,0], inputs[:,1], c=output_activity[:,out_idx],
                                     alpha=1., s=5, cmap=lcmap, vmin=vmin, vmax=vmax)
        # ax[out_idx].set_axis_off()
    # ax[3].set_axis_off()
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    #                                           cmap='Grays'),
    #              ax=ax[3],
    #              orientation='vertical',
    #              fraction=0.8)
    fig.show()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all(trainer, traindata, valdata, testdata, frozenFLAG, rolling_window=1, format='.png', save=True):
    train_results = trainer.trainlog
    val_results = trainer.validationlog
    test_results = trainer.testlog

    test_accuracy = 1. - test_results.classification_loss
    test_loss = test_results.cross_entropy_loss
    test_error = test_results.target_error.mean()

    os.makedirs(trainer.par.result_path + '/loss', exist_ok=True)
    os.makedirs(trainer.par.result_path + '/weights', exist_ok=True)
    os.makedirs(trainer.par.result_path + '/output', exist_ok=True)
    os.makedirs(trainer.par.result_path + '/control', exist_ok=True)
    os.makedirs(trainer.par.result_path + '/validation', exist_ok=True)
    os.makedirs(trainer.par.result_path + '/test', exist_ok=True)

    # Loss
    savepath = trainer.par.result_path + f'/loss/cross_entropy_loss_rw{rolling_window}' + format if save else None
    plot_loss(trainloss=train_results.cross_entropy_loss,
              valloss=val_results.cross_entropy_loss,
              testloss=test_loss,
              trainloss_std=0.,
              valloss_std=0.,
              testloss_std=0.,
              val_frequency=trainer.par.val_frequency,
              title=f'Test loss: {test_loss.mean():.2f} +/- {test_loss.std():.2f}',
              ylabel='Cross-entropy loss',
              rolling_window=rolling_window,
              legendFLAG=False,
              savepath=savepath)
    
    savepath = trainer.par.result_path + f'/loss/classification_accuracy_rw{rolling_window}' + format if save else None
    plot_loss(trainloss=1 - train_results.classification_loss,
              valloss=1 - val_results.classification_loss,
              testloss=test_accuracy,
              trainloss_std=0.,
              valloss_std=0.,
              testloss_std=0.,
              val_frequency=trainer.par.val_frequency,
              title=f'Test accuracy: {test_accuracy.mean() * 100:.1f}%',
              ylabel='Classification accuracy',
              rolling_window=rolling_window,
              legendFLAG=False,
              savepath=savepath)

    savepath = trainer.par.result_path + f'/loss/classification_accuracy_rw{50}' + format if save else None
    plot_loss(trainloss=1 - train_results.classification_loss,
              valloss=1 - val_results.classification_loss,
              testloss=test_accuracy,
              trainloss_std=0.,
              valloss_std=0.,
              testloss_std=0.,
              val_frequency=trainer.par.val_frequency,
              title=f'Test accuracy: {test_accuracy.mean() * 100:.1f}%',
              ylabel='Classification accuracy',
              rolling_window=50,
              legendFLAG=False,
              savepath=savepath)
    
    savepath = trainer.par.result_path + f'/loss/classification_accuracy_rw{20}' + format if save else None
    plot_loss(trainloss=1 - train_results.classification_loss,
              valloss=1 - val_results.classification_loss,
              testloss=test_accuracy,
              trainloss_std=0.,
              valloss_std=0.,
              testloss_std=0.,
              val_frequency=trainer.par.val_frequency,
              title=f'Test accuracy: {test_accuracy.mean() * 100:.1f}%',
              ylabel='Classification accuracy',
              rolling_window=20,
              legendFLAG=False,
              savepath=savepath)

    savepath = trainer.par.result_path + f'/loss/target_error_rw{rolling_window}' + format if save else None
    plot_loss(trainloss=train_results.target_error,
              valloss=val_results.target_error,
              testloss=test_error,
              trainloss_std=0.,
              valloss_std=0.,
              testloss_std=0.,
              val_frequency=trainer.par.val_frequency,
              title=f'Test target error: {test_error.mean():.2f} +/- {test_error.std():.2f}',
              ylabel='Target error [Hz]',
              rolling_window=rolling_window,
              legendFLAG=True,
              savepath=savepath)

    # Weights

    savepath = trainer.par.result_path + f'/weights/hidden_out_im' + format if save else None
    if frozenFLAG:    
        weights = train_results.weights_fine / trainer.par.n_pop
        # vmax = trainer.par.WMAX / trainer.par.n_pop
    else:
        weights = train_results.weights_hidden_out_fine / trainer.par.n_pop
        # vmax = trainer.par.WMAX_out / trainer.par.n_pop
    vmax = np.abs(weights.mean(axis=1)).max()
    plot_weights_hidden_out(weights=weights, epoch_step=20, savepath=savepath, vmin=-vmax, vmax=vmax)

    savepath = trainer.par.result_path + f'/weights/hidden_out_line' + format if save else None
    plot_weights_hidden_out_lines(weights=weights, savepath=savepath)

    savepath = trainer.par.result_path + f'/weights/hidden_out_line_dynapse' + format if save else None
    plot_weights_hidden_out_lines(weights=np.int64(np.rint(weights * trainer.par.n_pop)) / trainer.par.n_pop,
                                  savepath=savepath)

    savepath = trainer.par.result_path + f'/weights/hidden_out_change' + format if save else None
    plot_weight_changes(weights=weights * trainer.par.n_pop, savepath=savepath)

    savepath = trainer.par.result_path + f'/weights/hidden_out_diff' + format if save else None
    if frozenFLAG:
        weights = np.zeros_like(train_results.weights_fine)
        weights[1:,:,:,:] = (train_results.weights_fine[1:,:,:,:] - train_results.weights_fine[:-1,:,:,:]) / trainer.par.n_pop
    else:
        weights = np.zeros_like(train_results.weights_hidden_out_fine)
        weights[1:,:,:,:] = (train_results.weights_hidden_out_fine[1:,:,:,:] - train_results.weights_hidden_out_fine[:-1,:,:,:]) / trainer.par.n_pop
    vmax = np.abs(weights.mean(axis=1)).max()
    plot_weights_hidden_out(weights=weights, epoch_step=20, savepath=savepath, vmin=-vmax, vmax=vmax)
    
    if not frozenFLAG:
        savepath = trainer.par.result_path + f'/weights/in_hidden_im' + format if save else None
        weights = train_results.weights_in_hidden_fine / trainer.par.n_pop_hidden
        # vmax = trainer.par.WMAX_hidden / trainer.par.n_pop_hidden
        vmax = np.abs(weights.mean(axis=1)).max()
        plot_weights_in_hidden(weights=weights, epoch_step=20, savepath=savepath, vmin=-vmax, vmax=vmax)

        savepath = trainer.par.result_path + f'/weights/in_hidden_line' + format if save else None
        plot_weights_in_hidden_lines(weights=weights, savepath=savepath)

        savepath = trainer.par.result_path + f'/weights/in_hidden_line_dynapse' + format if save else None
        plot_weights_in_hidden_lines(weights=np.int64(np.rint(weights * trainer.par.n_pop_hidden)) / trainer.par.n_pop_hidden,
                                     savepath=savepath)

        # Feedback weights
        savepath = trainer.par.result_path + f'/weights/feedback_im' + format if save else None
        weights = train_results.weights_feedback / trainer.par.n_pop
        vmax = np.abs(weights.mean(axis=1)).max()
        plot_weights_hidden_out(weights=weights, epoch_step=20, savepath=savepath, vmin=-vmax, vmax=vmax)

        savepath = trainer.par.result_path + f'/weights/feedback_line' + format if save else None
        plot_weights_hidden_out_lines(weights=weights, savepath=savepath)


    # Test results
    predictions = np.argmax(trainer.testlog.outputs.mean(axis=1).mean(axis=1),axis=1)
    correct = np.sign(np.abs((predictions - testdata.labels.flatten())))

    savepath = trainer.par.result_path + f'/test/predictions' + format if save else None
    plot_predictions(inputs=testdata.inputs,
                     predictions=predictions,
                     input_frmin=trainer.par.input_frmin,
                     input_frmax=trainer.par.input_frmax,
                     savepath=savepath)
#                     input_min=trainer.par.input_frmin, input_max=trainer.par.input_frmax)

    savepath = trainer.par.result_path + f'/test/hidden_activity' + format if save else None
    plot_hidden_activity(inputs=trainer.testdata.inputs,
                         hidden_activity=trainer.testlog.hidden,
                         savepath=savepath)

    savepath = trainer.par.result_path + f'/test/output_feature_map' + format if save else None
    plot_output_feature_maps(inputs=testdata.inputs,
                            output_activity=trainer.testlog.outputs,
                            input_frmin=trainer.par.input_frmin,
                            input_frmax=trainer.par.input_frmax,
                            savepath=savepath)
    

    # Firing rates of different layers
    savepath = trainer.par.result_path + f'/output/outputs_training' + format if save else None
    plot_outputs(outputs=trainer.trainlog.outputs,
             labels=trainer.traindata.labels,
             target_frmin=trainer.par.target_frmin,
             target_frmax=trainer.par.target_frmax,
             rolling_window=100,
             savepath=savepath)

    savepath = trainer.par.result_path + f'/output/outputs_validation' + format if save else None
    plot_outputs(outputs=trainer.validationlog.outputs,
                labels=trainer.valdata.labels,
                target_frmin=trainer.par.target_frmin,
                target_frmax=trainer.par.target_frmax,
                rolling_window=100,
                savepath=savepath)

    savepath = trainer.par.result_path + f'/control/controls' + format if save else None
    plot_controls(cp=trainer.trainlog.control_p,
                cn=trainer.trainlog.control_n,
                labels=trainer.traindata.labels,
                out_idx=0,
                rolling_window=100,
                savepath=savepath)

    savepath = trainer.par.result_path + f'/control/control_summary' + format if save else None
    plot_control_summary(cp=trainer.trainlog.control_p,
                        cn=trainer.trainlog.control_n,
                        labels=trainer.traindata.labels,
                        out_idx=0,
                        rolling_window=100,
                        savepath=savepath)
    
    # TODO add feedback current plot
    
    # Validation
    for i in range(trainer.par.n_val):
        savepath = trainer.par.result_path + f'/validation/predictions_{i}' + format if save else None
        predictions = np.argmax(trainer.validationlog.outputs[i,:,:,:].mean(axis=1),axis=1)
        plot_predictions(inputs=valdata.inputs,
                        predictions=predictions,
                        input_frmin=trainer.par.input_frmin,
                        input_frmax=trainer.par.input_frmax,
                        savepath=savepath)

        if not frozenFLAG:
            savepath = trainer.par.result_path + f'/validation/hidden_activity_{i}' + format if save else None
            plot_hidden_activity(inputs=valdata.inputs,
                            hidden_activity=trainer.validationlog.hidden[[i],:,:,:],
                            savepath=savepath)
    