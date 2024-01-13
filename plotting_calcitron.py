import numpy as np

from matplotlib import pyplot as plt
import json

def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        loaded_dict = json.load(f)

    loaded_dict["param_dict"] = eval(loaded_dict["param_dict"].replace("inf", "np.inf"))
    return loaded_dict

def plot_accuracy(file_name):
    data = load_json(file_name)
    accuracies = data["accuracy_list"]
    plt.plot(accuracies)

plot_accuracy(r'CapacityPerceptron_Calcitron/modified_shouval_array_linear/P_200N_100num_epochs_10.json')
plt.show()
