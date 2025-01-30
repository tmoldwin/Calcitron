import numpy as np
from matplotlib import pyplot as plt
import constants
import param_helpers
import pattern_generators as pg
from calcitron import Calcitron
import supervisors as cs
import plasticity_rule as PR
from plasticity_rule import Region
import rule_comparison_grid as rcg
import plot_helpers as ph

plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)  # fontsize of the tick labels

eps = 0.05
num_rules = 4
all_coeffs = [[0, 0, 0.2, 1],
 [0.35, 0, 0.06, 1],
 [0, 0, 0, 1],
 [0.3, 0, 0, 1]]
# all_coeffs = [[0, 0, 0.2, 1],
#  [0., 0, 0.06, 1],
#  [0, 0, 0, 1],
#  [0.3, 0, 0, 1]]
all_thetas = np.array([[0.5, 1], [0.5, 0.7], [0.5, 1], [0.5, 0.7]])

eta = 0.02
def rule_from_thetas(thetas):
    return PR.Plasticity_Rule([Region('N', (-np.inf, thetas[0]), 0.5, 0),
                              Region('D', (thetas[0], thetas[1]), 0, eta),
                              Region('P', (thetas[1], np.inf), 0.5, eta)])

rules = [rule_from_thetas(thetas) for thetas in all_thetas]


rule_names = ['P supervisor\n(Global)', 'P supervisor\n(Targeted)', 'P&D supervisors\n(Global)', 'P&D supervisors\n(Targeted)']
input_bars = [0, 0, 0, 1, 1, 1]
output_bars = [1, 2, 3, 1, 2, 3]
ZDs = [0,0,all_thetas[2][0]+eps, all_thetas[3][0]+eps-all_coeffs[3][0]]
ZPs = [all_thetas[0][1]+eps,
       all_thetas[1][1]-all_coeffs[1][0],
       all_thetas[2][1]+eps,
       all_thetas[3][1]-all_coeffs[3][0]+eps]
all_Zs = [[ZPs[j] if i in (0,3) else ZDs[j] if i in (2,5) else 0 for i in range(6) ] for j in range(4)]
min_target = 1
max_target = 3
all_supervisors = [cs.homeostatic_supervisorPD(min_target=min_target, max_target = max_target, Z_p = ZPs[i], Z_d=ZDs[i]) for i in range(4)]
N = 20
P = 50
seed = 50

#patterns_targeted = [pg.pattern_gen(N, 1, density) for density in [0.15,0.8]]
evens = np.array([1 if number % 2 == 0 else 0 for number in range(1, N+1)])
odds = 1 - evens
patterns_targeted = [evens, odds]
weights_targeted = 0.05 * evens + 0.4 * odds
pattern_nums, inputs_targeted = pg.pattern_mixer(patterns_targeted, 80, seed = seed)
inputs_global = np.vstack((pg.pattern_gen(N, int(P/2), 0.15) ,pg.pattern_gen(N, int(P/2), 0.5)))
(inputs_global)
local_inputs = [inputs_targeted, inputs_targeted, inputs_targeted, inputs_targeted]
all_bars = [[input_bars, output_bars, all_Zs[i]] for i in range(num_rules)]
# (all_Zs)
# bar_names = [[r"$\hat{y}$" + f"={output_bars[i]}\n"
#                               + r"$\mathregular{x_{i}}$="
#                               + f"{input_bars[i]}"
#                               + f"\nZ={np.round(all_Zs[j][i],2)}" for
#                               i in range(6)] for j in range(num_rules)]
# (bar_names)
bar_names = [[(r"$\hat{y}=$" if i == 0 else "")
              + f"{output_bars[i]}\n"
              + (r"$\mathregular{x_{i}}$=" if i == 0 else "")
              + (f"{input_bars[i]}\n")
              + ("Z=" if i == 0 else "")
              +(f"{np.round(all_Zs[j][i],2)}")
              for i in range(6)] for j in range(num_rules)]
calcitrons = [Calcitron(all_coeffs[i],
                        rules[i],
                        supervisor=all_supervisors[i],
                        activation_function='linear') for i in range(len(rules))]

fig, bar_subplots, results_subplots = rcg.rule_comparison_grid(calcitrons, rule_names, all_bars, bar_names,
                                                               local_inputs, coeffs_to_use=[0, 2, 3], show_supervisor=1,
                                                               w_init=weights_targeted, plot_cbar=[3], figsize = (8.2, 8))
for ax in bar_subplots:
    ax.set_xticklabels(labels = ax.get_xticklabels(), fontsize = 7)
    xtick_positions = ax.get_xticks()
    xtick_labels = ax.get_xticklabels()
    # Set the alignment for the first xtick label to left
    xtick_labels[0].set_ha('right')
    ax.set_xticklabels(xtick_labels)

fig.axes[0].legend(bbox_to_anchor = (-0.18, 1.5))

# pattern_labels = 'E'
# for ax in np.array(results_subplots).ravel():
#     ax.set_xticks(range(len(pattern_nums)), labels = ['O' if pattern_num else 'E'  for pattern_num in pattern_nums], fontsize = 4)

letters = ['B','C','E','F']
labels = np.array([[letter + str(j) for letter in letters] for j in range(1,8)]).ravel()
ph.label_panels(fig, labels = labels, size = 8)
plt.savefig(constants.PLOT_FOLDER + '7.svg', dpi=fig.dpi)
plt.show()

param_helpers.fig_params(calcitrons, letters, 7)