import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable


def HebbSweep(pre, post, theta_d = 0.5, theta_p = 0.82, stepsize = 0.01, ax = None):
    alphas = np.linspace(0, 1.5 * theta_p, int(1.5 * theta_p / stepsize))
    gammas = np.linspace(0, 1.5 * theta_p, int(1.5 * theta_p / stepsize))
    mat = [[pre * alpha + post * gamma for alpha in alphas] for gamma in gammas]

    # Create plasticity matrix
    plasticity_mat = [[0 if val < theta_d else 1 if theta_d < val < theta_p else 2 for val in row] for row in mat]

    # Create a colormap
    cmap = mcolors.ListedColormap(['white', 'blue', 'red'])

    # Plot the plasticity matrix
    ax.imshow(plasticity_mat, origin='lower', cmap=cmap)
    return plasticity_mat

def all_rules():
    plasticity_mats = []
    fig, axes = plt.subplots(1, 4)
    for i, prepost in enumerate([(1, 0), (0, 1), (1, 1)]):
        plasticity_mat = HebbSweep(prepost[0], prepost[1], ax=axes[i])
        plasticity_mats.append(plasticity_mat)

    num_alphas = len(plasticity_mats[0])
    num_gammas = len(plasticity_mats[0][0])
    tuple_mat = [[[plasticity_mats[matnum][i][j] for matnum in range(len(plasticity_mats))] for j in range(num_gammas)]
                 for i in range(num_alphas)]

    int_mat = [[int(''.join(map(str, tup)), 3) for tup in row] for row in tuple_mat]
    flat_list = [tuple(item) for sublist in tuple_mat for item in sublist]

    counter = collections.Counter(flat_list)

    print(counter)
    print(len(counter.keys()))
    num_vals = len(counter.keys())

    cmap = plt.cm.get_cmap('nipy_spectral', num_vals)  # We have 3^3 = 27 unique tuples
    im = axes[3].imshow(int_mat, origin='lower', cmap=cmap)

    # Create a dictionary to map numbers to letters
    num_to_letter = {0: 'N', 1: 'D', 2: 'P'}

    # Create a colorbar
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(2*np.arange(num_vals), labels = counter.keys())  # Set tick positions

    plt.show()

all_rules()
