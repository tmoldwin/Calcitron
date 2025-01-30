import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import constants
SAVE_FOLDER = 'Calc_Plots\\'


def label_panels(fig, labels = None, xloc=0, yloc=1.0, size=constants.PANEL_LABEL_SIZE):
    if labels is None:
        labels = [str(x) for x in range(len(fig.axes))]
    for n in range(len(labels)):
        # label physical distance to the left and up:
            ax = fig.axes[n]
            trans = transforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
            ax.text(xloc, yloc, labels[n], transform=ax.transAxes + trans,
                    fontsize=size, va='bottom')

def label_panels_mosaic(fig, axes, xloc = 0, yloc = 1.0, size = 20):
    for key in axes.keys():
    # label physical distance to the left and up:
        ax = axes[key]
        trans = transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(xloc, yloc, key, transform=ax.transAxes + trans,
                fontsize=size, va='bottom')

def share_axes(subplot_array, sharex, sharey, delete_row_ticklabels = 1, delete_col_ticklabels = 1, delete_x_labels = 1, delete_y_labels = 1, one_d_row = True):
    shape_orig = np.array(subplot_array).shape
    subplot_array = np.atleast_2d(subplot_array)
    if len(shape_orig)==1 and one_d_row: #if it's a 1d array and we want it to be a row
        subplot_array = subplot_array.T #transpose it
    shape = subplot_array.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax = subplot_array[i,j]
            if sharex in ('rows', 'both'):
                if not ax.get_shared_x_axes().joined(ax, subplot_array[-1,j]):
                    ax.get_shared_x_axes().joined(ax, subplot_array[-1,j])
            if sharey in ('rows', 'both'):
                if not ax.get_shared_y_axes().joined(ax, subplot_array[-1,j]):
                    ax.get_shared_y_axes().joined(ax, subplot_array[-1,j])
            if sharex in ('cols', 'both'):
                if not ax.get_shared_x_axes().joined(ax, subplot_array[i,0]):
                    ax.get_shared_x_axes().joined(ax, subplot_array[i,0])
            if sharey in ('cols', 'both'):
                if not ax.get_shared_y_axes().joined(ax, subplot_array[i,0]):
                    ax.get_shared_y_axes().joined(ax, subplot_array[i,0])
            if delete_col_ticklabels and not(j==0):
                ax.set_yticklabels([])
            if delete_row_ticklabels and not (i == shape[0] - 1):
                ax.set_xticklabels([])
            if delete_y_labels and not (j == 0):
                ax.set_ylabel('')
            if delete_x_labels and not (i == shape[0] - 1):
                ax.set_xlabel('')










