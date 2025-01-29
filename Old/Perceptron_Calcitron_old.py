from calcitron import Calcitron
import numpy as np
from scipy.sparse import random as rand
from matplotlib import pyplot as plt
import json
import pathlib
from plasticity_rule import Region
from plasticity_rule import Plasticity_Rule



eta = 0.00004
#for asymptotic
f_d = 0.2
f_p = 1
#for linear:
# f_d = -1
# f_p = 1
#eta = 0.00001 - worked for p =100, N = 100 10000 epochs  and for p = 200, N = 100, 15000 epochs
#eta = 0.0001 - worked at P=550 , N=300

Plasticity_Rule([Region(name='N', bounds=(-np.inf, 0.5), fp=0.25, eta=0, fp_color='k', bar_code_color='w'),
                Region(name='D', bounds=(0.5, 0.6), fp=0.2, eta=4e-05, fp_color='b', bar_code_color='b'),
                Region(name='P', bounds=(0.6, 0.8), fp=1, eta=4e-05, fp_color='r', bar_code_color='r'),
                Region(name='PPNZ', bounds=(0.8, np.inf), fp=0.25, eta=0, fp_color='g', bar_code_color='g')])

# pr_critic = pr.Plasticity_Rule(eta_dict, FP_dict, name_dict=name_dict, color_dict=
#                                  {"N": "k", "D": "b", "P": "r", "PPNZ": "g"})

class Perceptron_Calcitron(Calcitron):

    def delta_bias(self, eta_b, y_hat_binary, y):
        if y_hat_binary == 1 and y == 0: #false positive
            return -eta_b
        elif y_hat_binary == 0 and y == 1: #false negative
            return eta_b
        else:
            return 0

    def multiple_epoch_training(self, num_of_epochs, X, y, protocol, param_dict,
                                a1=0.35, a2=0.55, plot= False, fixed_weights = 0,
                                include_z_in_output = False, eta_b = 0, early_stopping = 0, save = 0, **kwargs):
        P = len(X)
        epoch = 0
        weight_history = []
        bias_history = []
        accuracy_list = []
        converged = 0
        while converged == 0 and epoch < num_of_epochs:
            idx = np.arange(P)
            np.random.shuffle(idx)
            shuffled_X = np.array(X)[idx]
            shuffled_y = np.array(y)[idx]
            self.train(shuffled_X.copy(), shuffled_y.copy(), protocol,param_dict,
                                  a1 = 0.35, a2=0.55, plot= False,
                                  fixed_weights = 0, include_z_in_output = False, eta_b = eta_b, **kwargs)
            #weight_history.append(Calc_training_1.weight_history[-1,:])
            weight_history.append(self.weights.copy().tolist())
            bias_history.append(self.b)


            # exclusive or between y hat and Y target (Z)
            errors = np.bitwise_xor(shuffled_y.astype(int), self.y_hat_binary.astype(int))
            false_positive = sum(np.bitwise_and(errors, (self.y_hat_binary.astype(int) == 1)))
            false_negative = sum(errors) - (false_positive)

            ("false_positive", false_positive)
            ("false_negative", false_negative)
            num_of_errors = sum(np.bitwise_xor(shuffled_y.astype(int), self.y_hat_binary.astype(int)))
            error = num_of_errors / P
            accuracy = 1 - error
            accuracy_list.append(accuracy)
            (accuracy)
            if accuracy == 1 and early_stopping:
                converged = 1
            epoch += 1
        # plt.figure("accuracy")
        # plt.plot(accuracy_list)
        # plt.xlabel("epochs")
        # plt.ylabel("accuracy")
        # plt.figure("weights")
        # plt.plot(np.array(weight_history))
        # plt.figure("bias")
        # plt.plot(bias_history, linestyle = ':')
        if save:
            self.save_run(accuracy_list,bias_history,weight_history,P,protocol,param_dict)
        return accuracy_list, bias_history, weight_history

    def save_run(self,accuracy_list,bias_history,weight_history,P,protocol,param_dict):
        save_dict = {"accuracy_list":accuracy_list, "bias_history":bias_history, "weight_history":weight_history,
                     "P":P, "protocol":protocol.__name__, "param_dict":str(param_dict)}
        (save_dict)
        dir_name = "Capacity" + self.__class__.__name__ + '\\' + protocol.__name__ + "\\"
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        save_file_name = "P_"+str(P) + "N_"+str(N) + "num_epochs_"+str(len(accuracy_list))+ ".json"
        with open(dir_name + save_file_name, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # sparsity = 0.5
    # N = 200 #num of synapses
    # #P = 400 #num of patterns
    # X = np.array([np.ceil(rand(1,N, sparsity).A).flatten() for num in range(P)])
    ##another way for X:
    # x = np.zeros((1,100)) #a row in X
    # x[0:30,:30] = 1
    #  (x)
    # X = np.empty((P,N))
    # for num in range(P):
    #     # y=x[:]
    #     # (y)
    #     row_in_X = np.ceil(rand(1,100,0.3).A)
    #     X[num,:] = row_in_X
    # Y = np.squeeze(np.ceil(rand(1,P,sparsity).A)) #target

    # Z_matrix = [Z for ind in range(P)]
    #number of col - num of synapses, each col is one synapse
    #nu of rows = num of inputs
    #sparsity - or by making one with 30% then shuffling or by brenuli
    #make more columns than rows


    # bias_for_algo1 = -0.5*((N*sparsity*FP_dict[theta_d_perceptron_1,theta_p_perceptron_1])+
    #                       (N*sparsity*FP_dict[theta_p_perceptron_1,theta_ppnz_perceptron_1]))
    #
    # Calc_training_1 = Perceptron_Calcitron(0.45, 0, 0.1, 0.3, N, b=bias_for_algo1)
    # Calc_training_1.weights = np.array([random.uniform(0.7, 0.8) for w in range(N)])
    # Calc_training_1.multiple_epoch_training(10000,X, Y, protocol=fc.modified_shouval_array,
    #                       param_dict=perceptron_param_dict_for_generic_sbc,
    #                       a1=theta_d_perceptron_1, a2=theta_p_perceptron_1, eta_b = eta)

    P_list = [200]
    accuracy_mat = []
    bias_mat = []
    weight_history_mat = []
    sparsity = 0.5
    N = 100

    for p in P_list:
        X = np.array([np.ceil(rand(1, N, sparsity).A).flatten() for num in range(p)])
        Y = np.squeeze(np.ceil(rand(1, p, sparsity).A))
        bias_for_algo1 = -0.5*((N * sparsity * FP_dict[theta_d_perceptron_1, theta_p_perceptron_1]) +
                               (N * sparsity * FP_dict[theta_p_perceptron_1, theta_ppnz_perceptron_1]))

        Calc_training_1 = Perceptron_Calcitron([0.45, 0, 0.1, 0.3], pr_critic, N, b=bias_for_algo1)
        # Calc_training_1.weights = np.array([random.uniform(FP_dict[theta_d_perceptron_1, theta_p_perceptron_1],
        #                                                    FP_dict[theta_p_perceptron_1, theta_ppnz_perceptron_1]) for w in range(N)])
        # # accuracy_per_P, bias_history_per_P, weight_history_per_P = Calc_training_1.multiple_epoch_training(10,X, Y, protocol=fc.modified_shouval_array_linear,
        # #                      param_dict=perceptron_param_dict_for_generic_sbc,
        # #                     a1=theta_d_perceptron_1, a2=theta_p_perceptron_1, eta_b = eta, save = 1)
        # accuracy_mat.append(accuracy_per_P)
        bias_mat.append(bias_history_per_P)
        weight_history_mat.append(weight_history_per_P)



    plt.figure("capacity")
    for i in range(len(P_list)):
        plt.plot(accuracy_mat[i], label = "P = " + str(P_list[i]))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure("bias")
    for i in range(len(P_list)):
        plt.plot(bias_mat[i], label = "P = " + str(P_list[i]))
    plt.xlabel("Epochs")
    plt.ylabel("Bias")
    plt.legend()

    plt.figure("weights")
    for i in range(len(P_list)):
        plt.plot(weight_history_mat[i], label = "P = " + str(P_list[i]))
    plt.xlabel("Epochs")
    plt.ylabel("Weights")
    plt.show()
