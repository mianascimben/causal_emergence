''' 
This script contains functions usefull for the data visualization 
'''

import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import math

def plot_energy_landscape():
    pass

def plot_measures_subplots(
    data_1, 
    data_2=None, 
    CE_dict=None, 
    system_info=None,
    scale_labels=['micro', 'macro']
):

    list_of_measures = list(data_1.keys())
    
    # remove 'number of states'
    list_of_measures = [m for m in list_of_measures if m != "number of states"]

    if data_2 is None:
        all_data = data_1
    else: 
        all_data = {
            measure: [data_1[measure], data_2[measure]] 
            for measure in list_of_measures
        }

    n_plots = len(list_of_measures)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols) 
            
    plt.style.use('fast')
    plt.rcParams.update({'figure.autolayout': False})
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4*n_rows))
    axes = axes.flatten()

    # leave space above for title and info
    fig.subplots_adjust(top=0.80)

    for i, measure in enumerate(all_data):
        axes[i].bar(scale_labels, all_data[measure], width=0.7)
        axes[i].set_title(measure)
        axes[i].grid(True)
        
        if measure != 'EI':
            axes[i].set_xlim([0, 1.1])
            axes[i].set_ylim([0, 1.1])

    # remove unused axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    # title
    fig.suptitle("Causal Measures Across Scales", fontsize=18, y=0.98)

    # dynamic positions such that the two boxes we will create are alligned with the subplots
    left_ax = axes[0]
    right_ax = axes[1] if len(axes) > 1 else axes[0]
        
    pos_left = left_ax.get_position()
    pos_right = right_ax.get_position()

    # coordinats
    y_top = pos_left.y1 + 0.05   
    
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
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="whitesmoke",
                edgecolor="black"
            )
        )

    # -------------------------------
    # Right box: CE_dict
    # -------------------------------

    if CE_dict is not None:
        text_right = "\n".join(
            [f"${k}$: {v:.3f}" for k, v in CE_dict.items()]
        )

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

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
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
        if len(data) < 16:  
            
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
    

