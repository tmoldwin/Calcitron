import numpy as np
from matplotlib import pyplot as plt
import plasticity_rule as pr
import constants
import plot_helpers as ph
from plasticity_rule import Plasticity_Rule
from plasticity_rule import Region
theta_d = 0.45
theta_p = 0.85


regions_FPLR = [Region('N', (-np.inf, 0.5), 0.5, 0),
           Region('D', (0.5, 1), 0, 0.1),
           Region('P', (1, np.inf), 1, 0.1)]
regions_linear = [Region('N', (-np.inf, 0.5), np.nan, 0),
                  Region('D', (0.5, 1), np.nan, -0.1),
                  Region('P', (1, np.inf), np.nan, 0.2)]

FPLR = Plasticity_Rule(regions_FPLR, rule = 'FPLR')
Linear = Plasticity_Rule(regions_linear, rule='linear')

eta_dict_lin = {'N': 0, 'D': -0.1, 'P': 0.2}

mosaic = [['D', 'D', 'E', 'F', 'G'], ['H', 'I', 'J', 'K', 'L']]
labels = ['D', 'E', 'F', 'G','H','I','J','K','L','','','','']
fig, axes = plt.subplot_mosaic(mosaic, figsize=(8, 3.5), dpi = 300, constrained_layout=True)
Linear.dw_line_plot(ax = axes['D'], show_cbar=False)
annot_fs = 8
axes['D'].text(0.55, -0.08, 'Depression', size = annot_fs)
axes['D'].text(1.05, 0.15, 'Potentiation', size = annot_fs)
axes['D'].text(0.1, 0.01, 'No change', size=annot_fs)

# x_d_fill = np.arange(theta_d, theta_p, 0.01)
# x_p_fill = np.arange(theta_p, theta_p*1.5, 0.01)
# axes['D'].fill_between(x_d_fill, y2 = eta_dict_lin["D"] * np.ones_like(x_d_fill), y1 = np.zeros_like(x_d_fill), color = 'b')
# axes['D'].fill_between(x_p_fill, y2 = eta_dict_lin["P"] * np.ones_like(x_p_fill), y1 = np.zeros_like(x_p_fill), color = 'r')

Linear.dw_imshow(ax = axes['E'])
Linear.Ca_stim_plot(ax = axes['F'])
Linear.canonical_weight_change_from_Ca(ax = axes['G'])

FPLR.fp_and_eta_plot(ax = axes['H'])
FPLR.dw_line_plot(ax = axes['I'])
FPLR.dw_imshow(ax = axes['J'])
Linear.Ca_stim_plot(ax = axes['K'])
FPLR.canonical_weight_change_from_Ca(ax = axes['L'])
ph.label_panels_mosaic(fig, axes, size = 14)
#plt.tight_layout()
plt.savefig(constants.PLOT_FOLDER + '1.svg', dpi = fig.dpi)
#plt.show()
