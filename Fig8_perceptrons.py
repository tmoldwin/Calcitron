from matplotlib import pyplot as plt

import constants
import pattern_generators as pg
import numpy as np
import supervisors as sprvs
from plasticity_rule import Region
from plasticity_rule import Plasticity_Rule
from calcitron import Calcitron
import rule_comparison_grid as rcg
import random
import plot_helpers as ph
random.seed(1)
np.random.seed(1)

#
P = 6
sparsity = 0.25
N = 24
fp = 0
fd = 1
epochs = 10
eta = 0.2
X, y = pg.generate_perceptron_patterns(P, N, sparsity)

label_dict = {[tuple(x) for x in X][i]: y[i] for i in range(len(X))}
patterns = np.vstack([X for i in range(epochs)])
numbers = [i%P for i in range(len(patterns))]
print(patterns)

bias = -0.5 * ((N * sparsity * fp) +
                         (N * sparsity * fd))
target_rule = Plasticity_Rule([Region(name='N', bounds=(-np.inf, 0.5), fp=0.25, eta=0, fp_color='k', bar_code_color='w'),
                 Region(name='D', bounds=(0.5, 0.6), fp=0, eta=eta, fp_color='b', bar_code_color='b'),
                 Region(name='P', bounds=(0.6, 0.8), fp=1, eta=eta, fp_color='r', bar_code_color='r'),
                 Region(name='PPNZ', bounds=(0.8, np.inf), fp=0.25, eta=0, fp_color='g', bar_code_color='w')])
target_coeffs = [0.45, 0, 0.1, 1]
x_bar = ["local", "BAP", "$Z_P$",
         "BAP+\n$Z_P$",
         "local+\nBAP", "local+$Z_P$",
         "local\nBAP+$Z_P$"]
alpha_vector = np.array([1, 0, 0, 0, 1, 1, 1])
beta_vector = np.array([0, 0, 0, 0, 0, 0, 0])
gamma_vector = np.array([0, 1, 0, 1, 1, 0, 1])
delta_vector = np.array([0, 0, 0.3, 0.3, 0, 0.3, 0.3])
target_bar_matrix = [alpha_vector, beta_vector, gamma_vector, delta_vector]
target_calc = Calcitron(target_coeffs, target_rule, supervisor = sprvs.target_perceptron_supervisor(0.3, X,y), bias = bias)


#Section 2 critic perceptron
critic_rule = Plasticity_Rule(
    [Region(name='N', bounds=(-np.inf, 0.6), fp=0.25, eta=0, fp_color='k', bar_code_color='w'),
     Region(name='D', bounds=(0.6, 0.9), fp=0, eta=eta, fp_color='b', bar_code_color='b'),
     Region(name='P', bounds=(0.9, np.inf), fp=1, eta=eta, fp_color='r', bar_code_color='r')])

Z_d = 0.2
Z_p = 0.5
x_bar_critic_perceptron = ["local", r'$Z_D$', r'$Z_P$',
                            'local+\n' +r'$Z_D$', r'local' + '\n$Z_P$']
critic_coeffs = [0.45, 0, 0, 1]
cp_alpha_vector = np.array([1, 0, 0, 1, 1])
cp_beta_vector = np.zeros(len(x_bar_critic_perceptron))
cp_gamma_vector =  np.zeros(len(x_bar_critic_perceptron))
cp_Zd_vector = np.array([0, 1, 0, 1, 0]) * Z_d
cp_Zp_vector = np.array([0, 0, 1, 0, 1]) * Z_p
cp_Z_vector = cp_Zp_vector + cp_Zd_vector
critic_bar_matrix = [cp_alpha_vector, np.zeros(len(x_bar_critic_perceptron)), np.zeros(len(x_bar_critic_perceptron)), cp_Z_vector]
critic_perceptron_calc = Calcitron(critic_coeffs, critic_rule,
                                   supervisor=sprvs.critic_perceptron_supervisor(Z_d, Z_p, X, y), bias = bias)

calcitrons = [target_calc, critic_perceptron_calc]
titles = ["Target Perceptron", "Critic Perceptron"]
all_bar_mats = [target_bar_matrix, critic_bar_matrix]
x_barplot = [x_bar, x_bar_critic_perceptron]
all_local_inputs = [patterns,patterns]
fig, bar_subplots, results_subplots = \
    rcg.rule_comparison_grid(calcitrons, titles, all_bar_mats, x_barplot, all_local_inputs, show_supervisor=1, plot_cbar = [1], figsize = (8,8))
for ax in np.array(results_subplots).ravel():
    ax.set_xticks(range(len(patterns)), labels = [label_dict[tuple(pattern)] for pattern in patterns])
np.array(results_subplots)[5][0].set_xticks(range(len(patterns)), labels = [str(label_dict[tuple(patterns[i])]) + '\n' + str(numbers[i]) for i in range(len(patterns))])
np.array(results_subplots)[5][1].set_xticks(range(len(patterns)), labels = [str(label_dict[tuple(patterns[i])]) + '\n' + str(numbers[i]) for i in range(len(patterns))])
np.array(results_subplots)[5][0].set_xlabel('label\npattern #')
np.array(results_subplots)[5][1].set_xlabel('label\npattern #')
letters = ['B','D']
labels = np.array([[letter + str(j) for letter in letters] for j in range(1,8)]).ravel()
ph.label_panels(fig, labels = labels, size = 8)
plt.savefig(constants.PLOT_FOLDER + '8.svg', dpi=fig.dpi)
