''' 
This script contains functions usefull for the data visualization 
'''

import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import math

def dynamic_positions(axes, dy):
    # transform axes in a list 
    axes = np.atleast_1d(axes)

    left_ax = axes[0]
    right_ax = axes[1] if len(axes) > 1 else axes[0]
        
    pos_left = left_ax.get_position()
    pos_right = right_ax.get_position()

    # coordinats
    y_top = pos_left.y1 + dy 
    
    return  pos_left, pos_right, y_top


def plot_diagram_phase(temps, alphas, phase_data, mappings, 
                       title='***Missing Title***', 
                       cbar_label='', x_label='', y_label='',
                       vmin=None, vmax=None):
    fig, ax = plt.subplots()
    
    cmap = plt.cm.viridis  # puoi cambiare colormap
    norm = plt.Normalize(temps.min(), temps.max())
    
    # include the label for the micro system data
    mapping_label = mappings.copy()
    mapping_label.insert(0, 'None') 
    
    # index over the mappings
    for k in range(phase_data.shape[-1]):
        
        for i, T in enumerate(temps):
            color = cmap(norm(T))
            
            linestyle = ['-', '--', ':', '-.', '.'][k % 4]
            
            if i == 0:  # labelling the mapping 
                ax.plot(
                    alphas,
                    phase_data[i, :, k],
                    color=color,
                    linestyle=linestyle,
                    label = f'mapping: {mapping_label[k]}' 
                )
            else: 
                 ax.plot(
                     alphas,
                     phase_data[i, :, k],
                     color=color,
                     linestyle=linestyle
                 )

    # colorbar per riferimento
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if len(temps) > 1:
        plt.colorbar(sm, ax=ax, label=cbar_label)
    else: fig.text(
        0.75, 0.8,
        f'T = {temps}',
        ha="left",
        va="bottom",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="whitesmoke",
            edgecolor="black"
        )
    )
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    return fig, ax 
   
def heatmap_diagram_phase(x_data, y_data, z_data, system_info = None,
                       title='***Missing Title***', 
                       cbar_label='', x_label='', y_label='',
                       xlim=None, ylim=None):

    fig, ax = plt.subplots()
    ax.set_aspect('auto')
    
    if system_info is not None: fig.subplots_adjust(top=0.80)
    
    X, Y = np.meshgrid(x_data, y_data)

    pcm = ax.pcolormesh(X, Y, z_data, shading='auto')

    ax.set(xlim = xlim, ylim = ylim)
    # opzional: level line
    # ax.contour(X, Y, z_data, colors='black', linewidths=0.5)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # title
    ax.set_title(title, size=12, y=1.1)   
    
    pos_left, _, y_top = dynamic_positions(ax, dy = 0.02)
    
    if system_info is not None:
        text_left = "  |  ".join(
            [f"${k}$: {v}" for k, v in system_info.items()]
            )

        fig.text(
            pos_left.x0,      # alligned to the left subplot
            y_top,
            text_left,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="whitesmoke",
                edgecolor="black"
            )
        )
    return fig, ax

    
def plot_state_sequence_and_overlap(state_sequence, overlap_matrix, color_map="brg"):
    """
    Plots:
    1. Evolution of spin states over time (heatmap)
    2. Overlap with patterns over time (line plot)

    Args:
        state_sequence: numpy.ndarray of shape (steps, N) or (N, steps)
        overlap_matrix: numpy.ndarray of shape (nr_patterns, steps)
    """

    # shape is (steps, N)
    steps, N = state_sequence.shape
    nr_patterns, _ = overlap_matrix.shape

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Top: neuron states ---
    im = axs[0].imshow(
        state_sequence.T,
        aspect='auto',
        cmap='RdYlBu',
        interpolation='nearest'
        )
    axs[0].set_title("Spin states over time")
    axs[0].set_ylabel("Spin index")

    # Set minor ticks at cell boundaries
    axs[0].set_xticks(np.arange(0.5, steps, 1), minor=True)
    axs[0].set_yticks(np.arange(0.5, N, 1), minor=True)
    
    # Draw grid
    axs[0].grid(which='minor', color='grey', linestyle='-', linewidth=1)
    
    # Hide minor tick marks (keep only grid)
    axs[0].tick_params(which='minor', bottom=False, left=False)
    
    # Optional colorbar
    #plt.colorbar(im, ax=axs[0], orientation='vertical')
    
    # --- Bottom: overlaps ---
    for i in range(nr_patterns):
        axs[1].plot(np.arange(steps), overlap_matrix[i, :], label=f"$m^{i}(s(t))$")

    axs[1].set_title("Overlap measure in time")
    axs[1].set_ylim([-1.05, 1.05])
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("$m^{\mu}(s(t))$")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    

def plot_measures_subplots(
    df, 
    CE_df=None, 
    system_info=None,
    scale_labels=['first', 'second'], 
    title = ''
):
       
    # remove 'number of states'
    df = df.drop(columns=["number of states"], errors="ignore")
    
    list_of_measures = df.columns.tolist()
    scale_labels = df.index.tolist()
    
    n_plots = len(list_of_measures)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols) 
            
    plt.style.use('fast')
    plt.rcParams.update({'figure.autolayout': False})
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4*n_rows))
    axes = axes.flatten()

    # leave space above for title and info
    fig.subplots_adjust(top=0.70)

    for i, measure in enumerate(list_of_measures):
        values = df[measure].values
        
        axes[i].bar(scale_labels, values, width=0.7)
        axes[i].set_title(measure)
        axes[i].grid(True)
        
        if measure != 'EI':
            axes[i].set_ylim([0, 1.1])

    # remove unused axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    # title
    fig.suptitle(title, size=18, y=0.98)

    # dynamic positions such that the two boxes we will create are alligned with the subplots
    pos_left, pos_right, y_top = dynamic_positions(axes, dy = 0.05)
    
    # -------------------------------
    # Left box: system_info
    # -------------------------------
    if system_info is not None:
        text_left = "\n".join(
            [f"${k}$: {v}" for k, v in system_info.items()]
        )

        fig.text(
            pos_left.x0,      # alligned to the left subplot
            y_top,
            text_left,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="whitesmoke",
                edgecolor="black"
            )
        )

    # -------------------------------
    # Right box: CE_dict
    # -------------------------------

    if CE_df is not None:
        text_list = []
        for row in range(CE_df.shape[0]-1):
            text_row = "  ".join([f"${k}$: {v:.3f}" 
                                   for k, v in CE_df.iloc[row + 1].items()])
            text_list.append(text_row)
        text_right = '\n'.join(text_list)
        fig.text(
            pos_right.x1,    # alligned to the right subplot
            y_top,
            text_right,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightblue",
                edgecolor="black"
            )
        )
    return fig, axes


def heatmap(data, config_labels = None, title= '***Missing Title***', cbar_label = '', ax=None,
            cbar_kw=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array of shape (M, N).
        
    config_labels : np.ndarray, optional
        List of all configurations of the system used as labels on the aces.
        To guarantee readability this feature is implemented only for small 
        systems. The default is "None".
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
    plt.rcParams.update({'figure.autolayout': False})

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, aspect='auto')

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="baseline")

    # Show all ticks and label them with the respective list entries.
    if config_labels is not None and len(config_labels) < 16:  # Only for small systems
        ax.set_xticks(range(len(config_labels)), [str(c) for c in config_labels], rotation=45, fontsize=8)
        ax.set_yticks(range(len(config_labels)), [str(c) for c in config_labels], fontsize=8)
    elif config_labels is None: # when plotting the interaction matrix
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        
    ax.set_title(title)
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    if len(data) > 16 :
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.01)
    else: 
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im

def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                     textcolors=("white", "black"),
                     threshold=None):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
        
        # annotate the matrix only if its dimensions are small enough 
        if len(data) < 10:  
            
            # Normalize the threshold to the images color range.
            if threshold is not None:
                threshold = im.norm(threshold)
            else:
                threshold = im.norm(data.max())/2.
        
            # Set default alignment to center.
            kw = dict(horizontalalignment="center",
                      verticalalignment="center")
        
        
            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        
            # Loop over the data and create a `Text` for each "pixel".
            # Change the text's color depending on the data.
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    if  data[i, j] != 0: 
                        text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    

