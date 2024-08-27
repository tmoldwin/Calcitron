import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.random import choice
import constants
import param_helpers
import supervisors
from calcitron_calcium_bar_charts import calcium_barplot
from calcitron import Calcitron
import plasticity_rule as pr
from plasticity_rule import Region
import plot_helpers as ph
from matplotlib.colors import ListedColormap

from pattern_generators import signal_noise_gen, pattern_gen



plt.rc('xtick', labelsize=8)  # fontsize of the tick labels

fig = plt.figure(constrained_layout=True, figsize=(8, 7), dpi = 300)
fig_gs = GridSpec(5, 3, figure=fig, width_ratios=[0.5, 1, 0.04])
example_axes = fig.add_subplot(fig_gs[0:2, 0])
bar_axes = fig.add_subplot(fig_gs[2:5, 0])
results_axes = {'input':fig.add_subplot(fig_gs[0, 1]),
                'output': fig.add_subplot(fig_gs[1, 1]),
                'C': fig.add_subplot(fig_gs[2, 1]),
                'bar_code': fig.add_subplot(fig_gs[3, 1]),
                'w': fig.add_subplot(fig_gs[4, 1])}
# cbar_axes = {'input': fig.add_subplot(fig_gs[0, 2]), 'C':fig.add_subplot(fig_gs[2, 2]),
#              'bar_code':fig.add_subplot(fig_gs[3, 2]), 'w':fig.add_subplot(fig_gs[4, 2])}

wmin = 1.0
wmax = 3
epsilon = 0.1
N = 12
P = 60
density = 0.4
k = int(N * density)
alpha = 2
beta = 0.2
bias = -k * (wmin + (wmax-wmin)/2)
eta = 0.025
signal_prob = 0.4
theta_d = round((beta * k) - epsilon,1)
theta_p = round(theta_d + alpha - epsilon,1)
seed = 2
signal, labels, local_inputs = signal_noise_gen(N, P, signal_prob=signal_prob, density=density, seed = seed)
exemplars = np.array([signal, pattern_gen(N, 1, density=density, seed = None),
                      pattern_gen(N, 1, density=density, seed = None),signal, pattern_gen(N, 1, density=density, seed = None)])
example_axes.imshow(exemplars.T, aspect = 'auto', origin = 'lower', cmap = ListedColormap(['w','k']))
example_axes.set_xticks(range(len(exemplars)), labels = ['Signal', 'Noise', 'Noise','Signal','Noise'])
example_axes.set_ylabel(r'$\bf{Input}$' + '\nsyn #')

x_barplot = ['Inactive', 'Active']
alpha_vector = np.array([0,1])
beta_vector = np.array([np.sum(signal), np.sum(signal)])
bar_matrix = [alpha_vector, beta_vector]

rule = pr.Plasticity_Rule(regions=[Region('N', (-np.inf, theta_d), 0.5, 0),
                                   Region('D', (theta_d, theta_p), wmin, eta),
                                   Region('P', (theta_p, np.inf), wmax, eta)])
coeffs = [alpha, beta, 0,0]
calcium_barplot(bar_matrix, coeffs, rule, x_barplot, used_coeff_inds= [0,1], ax=bar_axes)
bar_axes.legend(loc="center left")

plot_dict = ["x", "weights", "output", "C_total", "Ca_bar_codes"]

rng = np.random.default_rng(seed)
w_init = (-bias / k * np.ones(N)) + rng.standard_normal(N)
hetero_calc = Calcitron([alpha, beta, 0, 0], rule, supervisor = supervisors.signal_noise_supervisor(signal),
                        bias = bias)
hetero_calc.train(local_inputs, w_init=w_init)

ph.share_axes(list(results_axes.values()), 'both', 'False')
hetero_calc.all_plots(axes=results_axes)

tick_labels = ['N' if val == 0 else 'S' for val in labels]
for ax in list(results_axes.values()):
    ax.set_xticks(range(P), labels = tick_labels, fontsize = 6)
# results_axes['output'].set_ylim([0.9*min(y_hat), 1.1*max(y_hat)])

ph.label_panels(fig, labels=['A', 'B', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], size=12)
plt.savefig(constants.PLOT_FOLDER + '5.svg', dpi=fig.dpi)
plt.savefig(constants.PAPER_PLOT_FOLDER + '5.tiff', dpi = fig.dpi)
param_helpers.fig_params([hetero_calc], ['All'], 5)
plt.show()

