import numpy as np
from matplotlib import pyplot as plt
from plasticity_rule import Plasticity_Rule as PR
from plasticity_rule import Region
import plot_helpers as ph
import rule_comparison_grid as rcg
from calcitron import Calcitron
import constants
import param_helpers

#plt.rc('xtick', labelsize=12)  # fontsize of the tick labels


bias = -1.8
N = 10
P = 50
prob = 0.4
k = int(N*prob)
eta = 0.1
seed = 99
rng = np.random.default_rng(seed)

hebb_regions = [Region('N', (-np.inf, 0.5), 0, 0),
                  Region('D', (0.5, 0.8), 0, eta),
                  Region('P', (0.8, np.inf), 1, eta)]
anti_hebb_regions = [Region('N', (-np.inf, 0.5), 0, 0),
                     Region('P', (0.5, 0.8), 1, eta),
                     Region('D', (0.8, np.inf), 0, eta)]
titles = ['Fire together wire together',
          'Fire together wire together & \nOut of sync lose your link',
          'Fire together lose your link',
          'Fire together lose your link & \nOut of sync wire together']
all_rules = [PR(hebb_regions), PR(hebb_regions), PR(hebb_regions), PR(anti_hebb_regions)]
all_coeffs = [[0.4,0,0.45,0], [0.55,0,0.7,0], [0.4,0,0.3,0], [0.55,0,0.6,0]]
w_init = (bias / k * np.ones(N)) + rng.standard_normal(N)
calcitrons = [Calcitron(coeffs, rule, bias = bias, w_init = w_init) for rule, coeffs in zip(all_rules, all_coeffs)]

# Generate the random local inputs using the seeded random number generator
local_inputs = np.array(rng.random((P, N)) <= prob, dtype=int)
# Create the list of all local inputs
all_local_inputs = [local_inputs for i in range(4)]

x_barplot = ["pre", "post", "both"]
alpha_vector = np.array([1, 0, 1])
gamma_vector = np.array([0, 1, 1])
bar_matrix = [alpha_vector, gamma_vector]
all_bar_mats = [bar_matrix, bar_matrix, bar_matrix, bar_matrix]
all_bar_titles = [x_barplot,x_barplot,x_barplot,x_barplot]


fig, results_subplots, bar_subplots = \
    rcg.rule_comparison_grid(calcitrons, titles, all_bar_mats, all_bar_titles, all_local_inputs, coeffs_to_use=[0, 2], plot_cbar = [3])
print(bar_subplots)
fig.axes[0].legend(bbox_to_anchor = (-0.2, 1.1))
letters = ['A','B','C','D','E','F']
labels = np.array([[letter + str(j) for letter in letters] for j in range(1,5)]).T.ravel()
ph.label_panels(fig, labels = labels)
plt.show(block = True)

plt.savefig(constants.PLOT_FOLDER + '2.svg')
# plt.savefig(constants.PAPER_PLOT_FOLDER + '2.tiff', dpi = fig.dpi)

rule_titles = ['A1-F1', 'A2-F2', 'A3-F3', 'A4-F4']
param_helpers.fig_params(calcitrons, rule_titles, 2)







