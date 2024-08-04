import numpy as np
from matplotlib import pyplot as plt

import constants
import param_helpers
from calcitron import Calcitron
from calcitron_calcium_bar_charts import calcium_barplot
import plot_helpers as ph
from plasticity_rule import Plasticity_Rule as PR
from plasticity_rule import Region

# TINY_SIZE = 6
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=TINY_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('xtick', labelsize=9)  # fontsize of the tick labels

fig, axes = plt.subplots(4, 4, figsize = (8, 4*1.8), dpi = 300)#, sharex = True)
x_barplot = ["pre", "post", "both"]
alpha_vector = np.array([1,0,1])
gamma_vector = np.array([0,1,1])
bar_matrix = [alpha_vector, gamma_vector]

dicts = [{'alpha': 0.2, 'gamma': 0.2,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.1, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.1,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 1.3},
{'alpha': 0.45, 'gamma': 0.45,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8}]

eta = 1
def rules_from_dict(dicts):
    rules = []
    coeffs = []
    calcitrons = []
    for d in dicts:
        regions = [Region('N', (-np.inf, d['theta_D']), 0.5, 0),
                   Region('D', (d['theta_D'], d['theta_P']), 0, eta),
                   Region('P', (d['theta_P'], np.inf), 1, eta)]
        rules.append(PR(regions))
        coeffs.append([d['alpha'], 0, d['gamma'],0])
        calcitrons.append(Calcitron(coeffs[-1], rules[-1]))
    return rules, coeffs

rules, coeffs = rules_from_dict(dicts)



#for anti hebb
# pre_post_and_both_below_thetaD = {r'$\alpha$': 0.2, 'gamma': 0.2,'theta_D': 0.8, 'theta_P': 0.5}
# pre_post_below_and_both_over_thetaD = {r'$\alpha$': 0.3, 'gamma': 0.3,'theta_D': 0.8, 'theta_P': 0.5}
# pre_below_post_and_both_over_thetaD = {r'$\alpha$': 0.1, 'gamma': 0.6,'theta_D': 0.8, 'theta_P': 0.5}
# pre_over_post_below_both_over_thetaD = {r'$\alpha$': 0.6, 'gamma': 0.1,'theta_D': 0.8, 'theta_P': 0.5}
# pre_post_and_both_over_thetaD = {r'$\alpha$': 0.6, 'gamma': 0.6,'theta_D': 1.3, 'theta_P': 0.5}
# pre_post_below_thetaD_and_both_over_thetaP = {r'$\alpha$': 0.45, 'gamma': 0.45,'theta_D': 0.8, 'theta_P': 0.5}
# pre_below_thetaD_post_over_thetaD_both_over_thetaP = {r'$\alpha$': 0.3, 'gamma': 0.6,'theta_D': 0.8, 'theta_P': 0.5}
# pre_below_thetaD_post_and_both_over_thetaP = {r'$\alpha$': 0.3, 'gamma': 0.9,'theta_D': 0.8, 'theta_P': 0.5}
# pre_over_thetaD_post_below_thetaD_both_over_thetaP = {r'$\alpha$': 0.6, 'gamma': 0.3,'theta_D': 0.8, 'theta_P': 0.5}
# pre_post_over_thetaD_both_over_thetaP = {r'$\alpha$': 0.6, 'gamma': 0.6,'theta_D': 0.8, 'theta_P': 0.5}
# pre_over_thetaD_post_and_both_over_thetaP = {r'$\alpha$': 0.6, 'gamma': 0.9,'theta_D': 0.8, 'theta_P': 0.5}
# pre_over_thetaP_post_below_thetaD_both_over_thetaP = {r'$\alpha$': 0.9, 'gamma': 0.3,'theta_D': 0.8, 'theta_P': 0.5}
# pre_over_thetaP_post_over_thetaD_both_over_thetaP = {r'$\alpha$': 0.9, 'gamma': 0.6,'theta_D': 0.8, 'theta_P': 0.5}
# pre_post_and_both_over_thetaP = {r'$\alpha$': 0.9, 'gamma': 0.9,'theta_D': 0.8, 'theta_P': 0.5}

unravelled_axes = axes.ravel()
for i in range(len(dicts)):
    params = dicts[i]
    ax = unravelled_axes[i]
    ax.set_ylim([0, 2.1])
    calcium_barplot(bar_matrix, coeffs[i], rules[i], used_coeff_inds= [0,2], x_labels=x_barplot, ax=ax)

fig.delaxes(unravelled_axes[-1])
fig.delaxes(unravelled_axes[-2])
ax.legend(bbox_to_anchor=(1.1, 1.05))
#ph.share_axes(axes,'both','both',1,1,0,1)
plt.tight_layout()
# plt.savefig(constants.PLOT_FOLDER + '3.svg', dpi=fig.dpi)
# plt.savefig(constants.PAPER_PLOT_FOLDER + 'fig3.tiff', dpi = fig.dpi)
plt.show()

def panel_to_position(panel_number):
    # Subtract 1 from the panel_number to start indexing from 0
    row = panel_number // 4 + 1
    col = panel_number % 4 + 1
    return f'{row},{col}'

param_helpers.fig_params(rules, [panel_to_position(i) for i in range(len(rules))], 3, coeffs = coeffs)
