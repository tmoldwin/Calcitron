import numpy as np
from matplotlib import pyplot as plt

from Old import functions_from_Demystifying_Calcium as fdc

from calcitron import Calcitron
from supervisors import target_perceptron_supervisor
from supervisors import critic_perceptron_supervisor


label_perceptron = {r'$\alpha$':0.45, r'$\gamma$':0.1, r'$\delta$':0.3,'theta_d': 0.5, 'theta_p': 0.6, 'theta_ppnz':0.8}
eta = 0.0003

f_d = -1
f_p = 1

theta_d = label_perceptron["theta_d"]
theta_p = label_perceptron["theta_p"]
theta_ppnz = label_perceptron["theta_ppnz"]
eta_dict_label_perceptron= {(-np.inf,theta_d):0, (theta_d,theta_p):eta,
                (theta_p,theta_ppnz):eta, (theta_ppnz, np.inf): 0}
FP_dict_label_perceptron = {(-np.inf,theta_d):0.25, (theta_d,theta_p):f_d,
               (theta_p,theta_ppnz):f_p, (theta_ppnz,np.inf):0.25}
label_perceptron_param_dict = {"eta_dict":eta_dict_label_perceptron, "FP_dict":FP_dict_label_perceptron,"soft_threshold": 0}


plot_dict_label = ["x", "weights", "output", "C_total", "Ca_bar_codes"]

N = 100
P = 100
local_inputs = [np.random.randint(2, size=P) for n in range(N)]
init_weights_label = np.random.uniform(f_d, f_p+0.1, N)
#init_weights = np.linspace(0.1,1.1,N)
y_label = np.random.randint(2, size=P)
label_perceptron_calc = Calcitron(label_perceptron[r'$\alpha$'], 0, label_perceptron[r'$\gamma$'], label_perceptron[r'$\delta$'], N)
label_perceptron_calc.weights = init_weights_label.copy()

label_perceptron_calc.train(local_inputs, target_perceptron_supervisor(y_label, eta_b = eta), protocol=fdc.modified_shouval_array,
                            param_dict= label_perceptron_param_dict, plot = True, things_to_plot = plot_dict_label,
                            include_z_in_output = False, no_firing=0)
ax_label_perceptron = label_perceptron_calc.ax
bias_subplot = ax_label_perceptron[2].twinx()
bias_subplot.plot(label_perceptron_calc.bias_history)
#should we plot y target?
#bias_subplot.set_ylim(min(label_perceptron_calc.bias_history)-0.01, max(label_perceptron_calc.bias_history)+0.01)

#ax_label_perceptron[2].plot(label_perceptron_calc.bias_history,  color = "g")

plt.subplots_adjust(left=0.08, right=0.9, top=0.926, bottom=0.11, wspace = 0.2, hspace = 0.357)
plt.show()

eta_critic = 0.0003
critic_perceptron = {r'$\alpha$':0.45, r'$\delta$' + r'$Z_D$': 0.2, r'$\delta$' + r'$Z_P$': 0.5, 'theta_d': 0.6, 'theta_p': 0.9}
f_d_critic = 0.7
f_p_critic = 0.9
theta_d_critic = critic_perceptron['theta_d']
theta_p_critic = critic_perceptron['theta_p']
eta_dict_critic_perceptron= {(-np.inf,theta_d_critic):0, (theta_d_critic,theta_p_critic):eta_critic,
                (theta_p_critic,np.inf):eta}
FP_dict_critic_perceptron = {(-np.inf,theta_d):0.25, (theta_d,theta_p):f_d_critic,
               (theta_p,np.inf):f_p_critic}
critic_perceptron_param_dict = {"eta_dict":eta_dict_critic_perceptron, "FP_dict":FP_dict_critic_perceptron,"soft_threshold": 0}

plot_dict_critic = ["x", "weights", "output", "C_total", "Ca_bar_codes"]

N_critic = 100
P_critic = 100
init_weights_critic = np.random.uniform(f_d_critic, f_p_critic+0.1, N_critic)
y = np.random.randint(2, size=P_critic)
Z_d_critic = 1
Z_p_critic = 2.5
delta = 0.2
bias_critic = -0.5 * ((N * f_d_critic) +(N * f_p_critic))
critic_perceptron_calc = Calcitron(critic_perceptron[r'$\alpha$'],0, 0, delta, N_critic)
critic_perceptron_calc.weights = init_weights_critic.copy()

critic_perceptron_calc.train(local_inputs, critic_perceptron_supervisor(y,Z_d_critic,Z_p_critic), protocol=fdc.modified_shouval_array,
                 param_dict= critic_perceptron_param_dict, plot = True, things_to_plot = plot_dict_critic,
                 include_z_in_output = False, no_firing=0)
(critic_perceptron_calc.C_tot)
# ax_label_perceptron = label_perceptron_calc.ax
# bias_subplot = ax_label_perceptron[2].twinx()
# bias_subplot.plot(label_perceptron_calc.bias_history)
plt.show()


