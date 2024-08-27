import numpy as np
from matplotlib import pyplot as plt

import plot_helpers
from Centroid_finder import example_find_peaks, compute_distance_grid, find_peaks
from scipy.ndimage import gaussian_filter
from calcitron import Calcitron
from calcitron_calcium_bar_charts import calcium_barplot
from plasticity_rule import Plasticity_Rule as PR
from plasticity_rule import Region
import constants
import param_helpers

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14

#plt.ion()
def rules_from_dict(dicts):
    rules = []
    coeffs = []
    calcitrons = []
    for d in dicts:
        if d['theta_D'] < d['theta_P']:
            regions = [Region('N', (-np.inf, d['theta_D']), 0.5, eta = 0),
                       Region('D', (d['theta_D'], d['theta_P']), 0, eta = 1),
                       Region('P', (d['theta_P'], np.inf), 1, eta = 1)]
        else:
            regions = [Region('N', (-np.inf, d['theta_P']), 0.5, eta = 0),
                       Region('P', (d['theta_P'], d['theta_D']), fp = 1, eta = 1),
                       Region('D', (d['theta_D'], np.inf), fp = 0, eta = 1)]
        rules.append(PR(regions))
        coeffs.append([d['alpha'], 0, d['gamma'],0])
        calcitrons.append(Calcitron(coeffs[-1], rules[-1]))
    return rules, coeffs
def create_peak_annotations(peaks, theta_D, theta_P, special = None):
    annotations = {}
    for i in range(len(peaks[0])):
        x = peaks[0][i]
        y = peaks[1][i]
        rule = rules_from_dict([{'alpha': x, 'gamma': y, 'theta_D': theta_D, 'theta_P': theta_P}])[0][0]
        x_code = int(rule.bar_code_from_C(x))
        y_code = int(rule.bar_code_from_C(y))
        xy_code = int(rule.bar_code_from_C(x+y))
        annotation = rule.region_names[x_code] + rule.region_names[y_code] + rule.region_names[xy_code]
        annotations[(x, y)] = annotation
    return annotations

def plot_lines_with_annotations(ax, lines, annotations, theta_d, theta_p, line_colors, specials):
    # Define bounds of the plane
    x_min, x_max = 0, 1.2
    y_min, y_max = 0, 1.2

    x_grid = np.linspace(x_min, x_max, 200)

    # Plot the lines with specified colors
    for line in lines:
        a, b, c = line
        color = line_colors.get(tuple(line), 'black')
        if b != 0:
            y_vals = -(a * x_grid + c) / b
            ax.plot(x_grid, y_vals, color=color)
        else:
            x_val = -c / a
            ax.axvline(x=x_val, color=color)

    # Annotate peaks
    for (x, y), annotation in annotations.items():
        if annotation in special:
            annotation = f"{annotation}*"
        ax.annotate(annotation, (x, y), textcoords="offset points", xytext=(0,-5), ha='center', color='black')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_title(r'$\theta_D = %.1f, \theta_P = %.1f$' % (theta_d, theta_p))
    ax.grid(False)

def order_indices(key, order):
    return [order[char] for char in key]


# Define parameters
thetas = [[(0.3,1.0), (0.7,1.0)], [(1.0,0.3), (1.0,0.7)]]
fig_names = ['3', '3_1']
annotation_orders = [{'N': 0, 'D': 1, 'P': 2}, {'N': 0, 'P': 1, 'D': 2}]
specials = [['DDD', 'NNP'],['PPP', 'NND']]
mx = 1.2

for fig_num in range(len(fig_names)):
    threshold_sets = thetas[fig_num]
    annotation_order = annotation_orders[fig_num]
    special = specials[fig_num]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    all_dicts = [{} for _ in range(2)]
    linaxes = [axes[0,0], axes [3,2]] #axes for the line plot
    for threshold_num, threshold_set in enumerate(threshold_sets):

        theta_d, theta_p = threshold_set
        ax = linaxes[threshold_num]

        # Define line colors
        line_colors = {
            (1, 0, -theta_d): 'blue',
            (1, 0, -theta_p): 'red',
            (0, 1, -theta_d): 'blue',
            (0, 1, -theta_p): 'red',
            (1, 1, -theta_d): 'blue',
            (1, 1, -theta_p): 'red',
            (1, 0, 0): 'black',
            (0, 1, 0): 'black',
            (1, 0, -mx): 'black',
            (0, 1, -mx): 'black',
        }

        line_list = [list(key) for key in line_colors.keys()]
        peaks = example_find_peaks(line_list, grid_size = 100, sigma = 4.9, plot = False)
        annotations = create_peak_annotations(peaks, theta_d, theta_p)
        for (x, y), annotation in annotations.items():
            all_dicts[threshold_num][annotation] = {'alpha': round(x, 2), 'gamma': round(y, 2), 'theta_D': theta_d, 'theta_P': theta_p}
        # Sort peaks based on annotation order
        all_dicts[threshold_num] = all_dicts[threshold_num] = dict(sorted(all_dicts[threshold_num].items(), key=lambda item: order_indices(item[0], annotation_order)))

        #plot the lines with their annotations
        plot_lines_with_annotations(ax, line_list, annotations, theta_d, theta_p, line_colors, specials = special)

    #axes[0, 1].set_yticks([]); axes[0, 1].set_ylabel(''); axes[0, 1].tick_params(labelleft=False)

    dicts_to_plot = list(all_dicts[0].values())
    dicts_to_plot.append(all_dicts[1][special[1]])

    # Continue with the rest of the code
    eta = 1

    rules, coeffs = rules_from_dict(dicts_to_plot)
    x_barplot = ["pre", "post", "both"]
    alpha_vector = np.array([1,0,1])
    gamma_vector = np.array([0,1,1])
    bar_matrix = [alpha_vector, gamma_vector]

    unravelled_axes = axes.ravel()
    for threshold_num in range(0, len(dicts_to_plot) - 1):
        params = dicts_to_plot[threshold_num]
        ax = unravelled_axes[threshold_num + 1]
        ax.set_ylim([0, 2.5])
        calcium_barplot(bar_matrix, coeffs[threshold_num], rules[threshold_num], used_coeff_inds=[0, 2], x_labels=x_barplot, ax=ax, set_ylim=False)

    #do the last one
    params = dicts_to_plot[-1]
    ax = unravelled_axes[-1]
    ax.set_ylim([0, 2.5])
    calcium_barplot(bar_matrix, coeffs[-1], rules[-1], used_coeff_inds=[0,2], x_labels=x_barplot, ax=ax, set_ylim=False)


    unravelled_axes[1].legend(bbox_to_anchor = (0.07, 0.45))
    labels = labels = ["A"] + [f"B{i}" for i in range(1, 14)] + ["C", "D"]
    plot_helpers.label_panels(fig, labels, size = 16)
    plt.tight_layout()
    labels = ["A"] + [f"B{i}" for i in range(1, 14)] + ["C", "D"]
    param_helpers.fig_params(rules, [label for label in labels if not label in ['A', 'C']], fig_names[fig_num], coeffs = coeffs)

    plt.savefig(constants.PLOT_FOLDER + fig_names[fig_num] + '.svg', dpi=fig.dpi)
    plt.savefig(constants.PAPER_PLOT_FOLDER + fig_names[fig_num] + '.tiff', dpi = fig.dpi)
    plt.close(fig)