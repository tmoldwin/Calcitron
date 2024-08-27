import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import calcitron
import constants
import param_helpers
from calcitron_calcium_bar_charts import calcium_barplot
from matplotlib.colors import ListedColormap
import plot_helpers as ph
import plasticity_rule as pr
from plasticity_rule import Region

# TINY_SIZE = 10
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(constrained_layout=True, figsize=(8, 5), dpi = 300)
fig_gs = GridSpec(2,5, figure=fig)
bar_subplot = fig.add_subplot(fig_gs[0, :])
hm_subplots = [fig.add_subplot(fig_gs[1,i])for i in range(5)]

x_barplot = [r"$\mathregular{x_{i}}$" + "=1", r"$\mathregular{x_{i}}$" + "=2",
             r"$\mathregular{x_{i}}$" + "=3",
             r"$\hat{y}$"+ "=1", r"$\hat{y}$"+ "=2", r"$\hat{y}$"+ "=3",
             r"$\mathregular{x_{i}}$" + "=1\n& " + r"$\hat{y}$"+ "=1",
             r"$\mathregular{x_{i}}$" + "=2\n& " + r"$\hat{y}$"+ "=1",
             r"$\mathregular{x_{i}}$" + "=1\n& " + r"$\hat{y}$"+ "=3"]

alpha_vector = np.array([1,2,3,0,0,0,1,2,1])
gamma_vector = np.array([0,0,0,1,2,3,1,1,3])
bar_matrix = [alpha_vector, gamma_vector]

eta = 0.1
frequency_dep = {'alpha': 0.3, 'gamma': 0.3,'theta_d': 0.5, 'theta_p': 0.8}
rule = pr.Plasticity_Rule(regions=[Region('N', (-np.inf, 0.5), 0.5, 0),
                                   Region('D', (0.5, 0.8), 0, eta),
                                   Region('P', (0.8, np.inf), 1, eta)])
coeffs = [0.3, 0, 0.3, 0]
calcium_barplot(bar_matrix, coeffs, rule,
                used_coeff_inds=[0, 2],
                x_labels=x_barplot, ax=bar_subplot)
bar_subplot.legend(bbox_to_anchor = (0, 1.05))

def frequency_dep_imshow(alpha, gamma, x_array, y_hat_array, theta_d = 0.5, theta_p = 0.8, ax = None):
    x_alpha = alpha * x_array
    y_hat_gamma = gamma * y_hat_array
    mat_for_plot = [[x_i + y_i for x_i in x_alpha] for y_i in y_hat_gamma]
    calc_mat = np.array(mat_for_plot)
    mat_for_plot = np.zeros_like(calc_mat)
    mat_for_plot[calc_mat > theta_p] = 2
    mat_for_plot[calc_mat <= theta_d] = 0
    mat_for_plot[(calc_mat <= theta_p) * (calc_mat > theta_d)] = 1
    im = ax.imshow(np.array(mat_for_plot), cmap=ListedColormap(['white', 'blue', 'red']), aspect="equal",origin="lower", extent = [0,x_array[-1],0, y_hat_array[-1]])
    ax.set_xlabel(r"$\mathregular{x_{i}}$")
    ax.set_ylabel(r"$\hat{y}$")
    ax.set_title(r'$\alpha$' + "=" + str(alpha) + ", " + r'$\gamma$'+"="+str(gamma))
    return im

x = np.array(np.arange(0, 4.001, 0.01))
y_hat = np.array(np.arange(0, 4.001, 0.01))

frequency_dep_imshow(0.3, 0.0, x, y_hat, ax = hm_subplots[0])
frequency_dep_imshow(0.3, 0.15, x, y_hat, ax = hm_subplots[1])
frequency_dep_imshow(0.3, 0.3, x, y_hat, ax=hm_subplots[2])
frequency_dep_imshow(0.15, 0.3, x, y_hat, ax=hm_subplots[3])
im = frequency_dep_imshow(0, 0.3, x, y_hat, ax=hm_subplots[4])

cbar = plt.colorbar(im, label='Plasticity', ax=hm_subplots[-1], fraction=0.046, pad=0.04)
cbar.set_ticks([0.33, 0.99, 1.65], labels=rule.region_names)
ph.label_panels(fig, labels = ['A','B1','B2','B3','B4','B5'], size = 12)
plt.savefig(constants.PLOT_FOLDER + '4.svg', dpi=fig.dpi)
plt.savefig(constants.PAPER_PLOT_FOLDER + '4.tiff', dpi = fig.dpi)
#plt.show()

panel_labels = ['A','B1','B2','B3','B4','B5']
#first A then the examples
coeffs_mat = [[0.3,0,0.3,0], [0.3,0,0,0], [0.3,0,0.15,0],[0.3,0,0.3,0],[0.15,0,0.3,0],[0,0,0.3,0]]
param_helpers.fig_params([rule for i in range(len(panel_labels))], panel_labels, 4, coeffs = coeffs_mat)