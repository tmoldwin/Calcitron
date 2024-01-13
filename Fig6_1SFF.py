from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

import constants
import pattern_generators as pg
from calcitron import Calcitron
from calcitron_calcium_bar_charts import calcium_barplot
import numpy as np
from matplotlib import pyplot as plt
from supervisors import one_shot_flip_flop_supervisor
import plot_helpers as ph
import plasticity_rule as pr
from plasticity_rule import Region

# SMALL_SIZE = 10
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

alpha = 0.2
delta = 0.6
theta_d = 0.5
theta_p = 0.7
eta = 1

rule = pr.Plasticity_Rule(regions=[Region('N', (-np.inf, theta_d), 0.5, 0),
                                    Region('D', (theta_d, theta_p), 0, eta),
                                    Region('P', (theta_p, np.inf), 1, eta)])
coeffs = np.array([alpha, 0, 0, delta])

fig = plt.figure(constrained_layout=True, figsize=(8, 6.7), dpi = 300)
fig_gs = GridSpec(6, 2, figure=fig, width_ratios=[1, 3])
rule_col = 0
example_col = 0
bar_col = 0
result_col = 1
cbar_col = 2
rule_axes = [fig.add_subplot(fig_gs[i, rule_col]) for i in range(3)]
#example_ax = fig.add_subplot(fig_gs[4, example_col])
bar_ax = fig.add_subplot(fig_gs[3:, bar_col])
result_names = ['input','Z','C','bar_code','w','output']
results_axes = {key:fig.add_subplot(fig_gs[i, result_col]) for i, key in enumerate(result_names)}
# cbar_axes = {'input': fig.add_subplot(fig_gs[0, cbar_col]), 'C': fig.add_subplot(fig_gs[2, cbar_col]),
#              'bar_code': fig.add_subplot(fig_gs[3, cbar_col]), 'w': fig.add_subplot(fig_gs[4, cbar_col])}

rule.fp_and_eta_plot(ax=rule_axes[0])
Ca_stim = np.zeros(100)
Ca_stim[[20,50,70]] = np.array([theta_d, theta_p, theta_d]) + 0.1
rule.Ca_stim_plot(Ca_stim, line_colors = ['k'], ax = rule_axes[1])
rule.weight_change_plot_from_Ca(w0 = 0.5, Ca = Ca_stim, ax = rule_axes[2], line_colors=['k'], fp_colors=['k','b','r'], line_styles=['-'])

calcitron_x_bar = ["Active syn", "Plateau", "Active +\nplateau"]
alpha_vector = np.array([1,0,1])
delta_vector = np.array([0,1,1])
calcitron_mat = [alpha_vector, delta_vector]
calcium_barplot(calcitron_mat, coeffs, rule, calcitron_x_bar, used_coeff_inds=[0,3], ax=bar_ax)
bar_ax.legend(bbox_to_anchor = (0, 0.99))


N = 14
P = 4
density = 0.5
bias = -(density * N - 1)

patterns = pg.pattern_gen(N,P,density)
# example_ax.imshow(patterns.T, aspect='auto', origin='lower', cmap=ListedColormap(['w', 'k']))
# example_ax.set_xticks(range(len(patterns)), labels=range(len(patterns)))
input_nums = list(range(len(patterns)))*8
inputs = [patterns[val] for val in input_nums]
Z = np.zeros_like(input_nums)
one_inds = range(4,len(inputs),7)
Z = [1 if i in one_inds else Z[i] for i in range(len(Z))]

_1SFF = Calcitron([alpha, 0, 0, delta], rule, supervisor = one_shot_flip_flop_supervisor(y=Z), bias = bias)
_1SFF.train(inputs)
_1SFF.all_plots(axes = results_axes)
for ax in list(results_axes.values()):
    ax.set_xticks(range(len(input_nums)),input_nums)
ph.label_panels(fig, labels=['A1', 'A2', 'A3', 'B', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
fig.savefig(ph.SAVE_FOLDER+str(6) +'.svg', dpi = fig.dpi)
fig.savefig(constants.PAPER_PLOT_FOLDER+str(6) +'.tiff', dpi = fig.dpi)

plt.show()
