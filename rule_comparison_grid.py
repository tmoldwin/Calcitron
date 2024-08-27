import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from calcitron import Calcitron
from calcitron_calcium_bar_charts import calcium_barplot
import plot_helpers as ph


# SMALL_SIZE = 12
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 12
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

def rule_comparison_grid(calcitrons, rule_names, all_bar_mats, bar_names, all_inputs, coeffs_to_use=[0, 1, 2, 3],
                         plot_cbar=[], show_supervisor=0, w_init='middle', figsize = None):
    num_rules = len(calcitrons)
    for i,calc in enumerate(calcitrons):
        calc.train(all_inputs[i], w_init=w_init)
    max_calcium = max([np.max(calc.C_tot) for calc in calcitrons])
    all_rules = [calcitron.plasticity_rule for calcitron in calcitrons]
    all_coeffs = [calcitron.coeffs for calcitron in calcitrons]
    num_rows = 6 + int(show_supervisor)  # if there's a supervior, add another row
    if not figsize is None:
        fig = plt.figure(constrained_layout=True, figsize=figsize, dpi = 300)
    else:
        fig = plt.figure(constrained_layout=True, figsize=(8, (num_rows * 1.5)), dpi = 300)
    fig_gs = GridSpec(num_rows, num_rules + 1, figure=fig,
                      height_ratios=[2] + list(np.ones(num_rows - 1)),
                      width_ratios=list(np.ones(num_rules)) + [0.01])
    bar_subplots = [fig.add_subplot(fig_gs[0, j:j + 1]) for j in range(num_rules)]
    results_subplots = np.array([[fig.add_subplot(fig_gs[i,j]) for j in range(num_rules)] for i in range(1, num_rows)])
    ph.share_axes(results_subplots, 'both', 'cols', 1, 1, 1, 1, 1)
    # cbar_ax_1 = fig.add_subplot(fig_gs[1, -1])
    # cbar_ax_2 = fig.add_subplot(fig_gs[3, -1])
    # cbar_ax_3 = fig.add_subplot(fig_gs[4, -1])
    # cbar_ax_4 = fig.add_subplot(fig_gs[5, -1])
    # cbar_axes = {'input': cbar_ax_1, 'C': cbar_ax_2, 'bar_code': cbar_ax_3, 'w': cbar_ax_4}
    for col_num in range(num_rules):
        bar_ax = bar_subplots[col_num]
        coeffs = all_coeffs[col_num]
        rule = all_rules[col_num]
        bars_mat = all_bar_mats[col_num]
        if not show_supervisor:
            plots_ax = {"input": results_subplots[0, col_num],
                        "output": results_subplots[1, col_num],
                        "C": results_subplots[2, col_num],
                        "bar_code": results_subplots[3, col_num],
                        "w": results_subplots[4, col_num]}
        else:  # if there's a supervisor, add another row
            plots_ax = {"input": results_subplots[0, col_num], "Z": results_subplots[1, col_num],
                        "C": results_subplots[2, col_num], "bar_code": results_subplots[3, col_num],
                        "w": results_subplots[4, col_num], "output": results_subplots[5, col_num]}
        calcium_barplot(bars_mat, coeffs, rule, bar_names[col_num], used_coeff_inds=coeffs_to_use, ax=bar_ax)
        bar_ax.set_title(rule_names[col_num] + '\n' + bar_ax.get_title())
        calc = calcitrons[col_num]
        calc.all_plots(axes=plots_ax, plot_cbar=col_num in plot_cbar,
                        C_range=(0, max_calcium))
    ph.share_axes(results_subplots,'both','cols',1,1,1,1,1)
    ph.share_axes(bar_subplots,1,1,1,0,1,1,False)
    return fig, bar_subplots, results_subplots #, cbar_axes