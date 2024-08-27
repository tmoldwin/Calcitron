import numpy as np
from matplotlib import pyplot as plt
import constants
import plasticity_rule as pr

coeff_names = ['alpha', 'beta', 'gamma', 'delta']
color_dict = ["g", "tab:orange", "hotpink", "k"]
label_dict = [r'$C^i_{local}$', r'$C_{het}$', r'$C_{BAP}$', r'$C_{SPRV}$']

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def texify(strings):
    strings = np.atleast_1d(strings)
    greek = ['alpha', 'beta', 'gamma', 'delta', 'theta']
    greekdic = {x: '\\' + x for x in greek}
    new_strings = [r'$' + replace_all(string, greekdic) + '$' for string in strings]
    if len(new_strings) == 1:
        return new_strings[0]
    else:
        return new_strings

def calcium_barplot(binary_mat, coeffs, rule, x_labels, used_coeff_inds = [0, 1, 2, 3], ax=None,
                    rotation=0, set_ylim = True):
    '''
    '''
    (f'coeffs: {coeffs}', f'rule: {rule}', f'x_labels: {x_labels}', f'used_coeff_inds: {used_coeff_inds}')
    binary_mat = np.atleast_2d(binary_mat)
    y_list = []
    used_coeffs = [coeffs[i] for i in used_coeff_inds]
    used_coeff_names = [coeff_names[i] for i in used_coeff_inds]
    for num_row, value in enumerate(used_coeffs):
        y = used_coeffs[num_row] * binary_mat[num_row]
        y_list.append(y)
    bottom = np.zeros(len(binary_mat[0]))
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for i ,y in enumerate(y_list):
        ax.bar(x_labels, y, bottom= bottom,
               color = color_dict[used_coeff_inds[i]],
               label = label_dict[used_coeff_inds[i]])
        bottom += y
    bar_codes = rule.bar_code_from_C(bottom)
    annots = [rule.region_names[int(bc)] for bc in bar_codes]
    for i, region in enumerate(rule.regions[1:]):
        ax.axhline(y = region.bounds[0], c = region.fp_color, linestyle =':', linewidth = 3)
    ax.set_yticks(rule.thetas)
    ax.set_yticklabels(labels = rule.theta_names)
    ax.tick_params(axis = 'x', rotation=rotation)
    ax.set_ylabel('$\mathregular{C^i_{total}}$')
    if set_ylim:
        ax.set_ylim(0, 1.2 * np.max(np.hstack((bottom, rule.thetas))))
    for i in range(len(x_labels)):
        ax.annotate(annots[i], (i, 0.9*ax.get_ylim()[-1]), ha='center')
    title_string = ', '.join([texify(coeff) + ' = ' +
                              str(used_coeffs[i]) for i, coeff
                              in enumerate(used_coeff_names)])
    title_string += '\n' + ', '.join([rule.theta_names[i] + ' = ' + str(rule.thetas[i]) for i in range(len(rule.thetas))])
    ax.set_title(title_string)


