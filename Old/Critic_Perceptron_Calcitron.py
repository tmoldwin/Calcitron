from Old import functions_for_calcitron as fc
import numpy as np
from scipy.sparse import random as rand
import random
from matplotlib import pyplot as plt

from Old.Perceptron_Calcitron_old import Perceptron_Calcitron
eta = 0.000005
#eta = 0.00001 - for N=300 and P = 550

param_dict_bar2 = {'F_d': 0.7, 'F_p': 0.9, 'gamma': 0, 'Z_d': 0.2, 'Z_p': 0.5, 'alpha': 0.6, 'theta_d': 0.6,
                       'theta_p': 0.9}
theta_d_algo2 = 0.6
theta_p_algo2 = 0.9
eta_dict_SBC_algo2 = {(-np.inf,theta_d_algo2):0, (theta_d_algo2,theta_p_algo2):eta,
                (theta_p_algo2,np.inf):eta}
FP_dict_SBC_algo2 = {(-np.inf,theta_d_algo2):0.25, (theta_d_algo2,theta_p_algo2):0.7, (theta_p_algo2,np.inf):0.9}
perceptron_param_dict_for_generic_sbc = {"eta_dict":eta_dict_SBC_algo2, "FP_dict":FP_dict_SBC_algo2,"soft_threshold": 0}
f_d = FP_dict_SBC_algo2[theta_d_algo2,theta_p_algo2]
f_p = FP_dict_SBC_algo2[theta_p_algo2,np.inf]

class Critic_Perceptron_Calcitron(Perceptron_Calcitron):
    def __init__(self, alpha, beta, gamma, delta, N, Z_p, Z_d, b=0):
        super().__init__(alpha, beta, gamma, delta, N, b)
        self.Z_p = Z_p
        self.Z_d = Z_d
        self.name = "critic perceptron"
        #added Z_p and Z_d to the class so can use in the next functions

    def calculate_Z_critic(self, Z_i, y_hat_binary_i):
        if y_hat_binary_i == 1 and Z_i == 0: #false positive
            return self.Z_d
        elif y_hat_binary_i == 0 and Z_i == 1: #false negative
            return self.Z_p
        else:
            return 0
        #false positive - Z_d
        #false negative - Z_p

#N = 200 #num of synapses
#P = 550 #num of patterns
#X = np.array([np.ceil(rand(1,N, sparsity).A).flatten() for num in range(P)])
# Z = np.squeeze(np.ceil(rand(1,P,sparsity).A)) #target
# bias_for_algo2 = -0.5*((N*sparsity*FP_dict_SBC_algo2[theta_d_algo2,theta_p_algo2])+
#                       (N*sparsity*FP_dict_SBC_algo2[theta_p_algo2,np.inf]))
#
# Calc_training_2 = Critic_Perceptron_Calcitron(0.45, 0, 0, 1, N, b= bias_for_algo2, Z_p = 0.5, Z_d = 0.2)
# Calc_training_2.weights = np.array([random.uniform(f_d, f_p) for w in range(N)])
# Calc_training_2.multiple_epoch_training(10000,X, Z, protocol=fc.modified_shouval_array,
#                       param_dict=perceptron_param_dict_for_generic_sbc,
#                       a1=theta_d_algo2, a2=theta_p_algo2, eta_b = 0)

P_list = [100,200,300]
accuracy_mat_critic = []
bias_mat_critic = []
sparsity = 0.5
N = 100
for p in P_list:
    X = np.array([np.ceil(rand(1, N, sparsity).A).flatten() for num in range(p)])
    Z = np.squeeze(np.ceil(rand(1, p, sparsity).A))  # target
    bias_for_algo2 = -0.5 * ((N * sparsity * FP_dict_SBC_algo2[theta_d_algo2, theta_p_algo2]) +
                             (N * sparsity * FP_dict_SBC_algo2[theta_p_algo2, np.inf]))

    Calc_training_2 = Critic_Perceptron_Calcitron(0.45, 0, 0, 1, N, b=bias_for_algo2, Z_p=0.5, Z_d=0.2)
    Calc_training_2.weights = np.array([random.uniform(f_d, f_p) for w in range(N)])
    accuracy_per_P_critic, bias_history_per_P_critic = Calc_training_2.multiple_epoch_training(10000, X, Z, protocol=fc.modified_shouval_array,
                                            param_dict=perceptron_param_dict_for_generic_sbc,
                                            a1=theta_d_algo2, a2=theta_p_algo2, eta_b=eta)
    accuracy_mat_critic.append(accuracy_per_P_critic)
    bias_mat_critic.append(bias_history_per_P_critic)

plt.figure("critic capacity")
for i in range(len(P_list)):
        plt.plot(accuracy_mat_critic[i], label = "P = " + str(P_list[i]))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


plt.figure("bias_critic")
for i in range(len(P_list)):
    plt.plot(bias_mat_critic[i], label="P = " + str(P_list[i]))
plt.xlabel("Epochs")
plt.ylabel("Bias")
plt.legend()
#plt.show()