import numpy as np


def omega(C, param_dict):
    k_d, k_p, theta_d, theta_p, b1, b2 = param_dict["k_d"], param_dict["k_p"], param_dict["theta_d"], param_dict[
        "theta_p"], \
                                         param_dict["b1"], param_dict["b2"]
    return -k_d / (1 + np.e ** (-b2 * (C - theta_d))) + (k_d + k_p) / (1 + np.e ** (-b1 * (C - theta_p)))

def step_function(x, dict):
    for interval in dict.keys():
        if x >= interval[0] and x < interval[1]:
            return dict[interval]


def shouval1(C, w, param_dict):
    hard_threshold = param_dict["hard_threshold"]
    eta = param_dict["eta"]
    if hard_threshold == 1:
        return eta * hard_omega(C, param_dict)
    elif hard_threshold == 2:
        return eta * shouval_omega(C, param_dict)
    else:
        return eta * omega(C, param_dict)



def modified_shouval_math(Ca, w, param_dict):
    eta_dict, FP_calcium_dict, soft_threshold = param_dict["eta_dict"], param_dict["FP_calcium_dict"], param_dict[
        "soft_threshold"]
    if soft_threshold == 1:
        return soft_omega(Ca, param_dict['p1']) * (soft_omega(Ca, param_dict['p2']) - w)
    else:
        return step_function(Ca, eta_dict) * (step_function(Ca, FP_calcium_dict) - w)


def modified_shouval_array(Ca, weights, param_dict, N):
    delta_w = np.zeros(N)
    eta_dict, FP_dict, soft_threshold = \
        param_dict["eta_dict"], param_dict["FP_dict"], param_dict["soft_threshold"]
    for interval in eta_dict.keys():
        is_in_interval = np.bitwise_and((np.heaviside(Ca - interval[0], 0)).astype(int),
                                        (np.heaviside(interval[1] - Ca, 0).astype(int)))
        is_in_interval_updated = eta_dict[interval] * is_in_interval * (FP_dict[interval] - weights)
        delta_w += is_in_interval_updated
    return delta_w

functions = [shouval1, modified_shouval_math]


def calculate_plasticity(C, weight, protocol, param_dict, dif_func_list, just_protocol = False):
    if dif_func_list == 1:
        delta_w = GB_func_list[protocol](C, weight, param_dict)
    elif just_protocol:
        delta_w = protocol(C, weight, param_dict)
    else:
        delta_w = functions[protocol](C, weight, param_dict)
    return delta_w


def run_plasticity(C, weight_0, protocol, param_dict, dif_func_list, just_protocol = False):
    weight_over_time = np.zeros(len(C))
    weight_over_time[0] = weight_0
    for i in range(len(C) - 1):
        delta_w = calculate_plasticity(C[i], weight_over_time[i], protocol, param_dict, dif_func_list, just_protocol=just_protocol)
        weight_over_time[i + 1] = weight_over_time[i] + delta_w
    return weight_over_time

def w_graph_different_fps(subgraph, loc, C_range_potentiation, C_range_depression, start_weight, fp_min, fp_max, protocol, duration,
                           param_dict, dif_func_list=False, no_row_col = False, no_fp = 0):
    if no_row_col == True:
        subgraph.plot(range(len(C_range_potentiation)),
                                cf.run_plasticity(C_range_potentiation, fp_min, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "r", linestyle=":")
        subgraph.plot(range(len(C_range_potentiation)),
                                cf.run_plasticity(C_range_potentiation, start_weight, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "r")
        subgraph.plot(range(len(C_range_potentiation)),
                                cf.run_plasticity(C_range_depression, fp_max, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "b", linestyle=":")
        subgraph.plot(range(len(C_range_potentiation)),
                                cf.run_plasticity(C_range_depression, start_weight, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "b")
        subgraph.set_xticks([0,duration])
        subgraph.set_xticklabels(['stim' + '\n' + 'start','stim' + '\n' + 'end'], fontsize = 8)
        # subgraph[row,col].set_ylabel('w')
        subgraph.set_xlabel('Time (hours)', labelpad = -10)
        #subgraph.legend(fontsize=9)
    if no_fp:
        subgraph[loc].plot(range(len(C_range_potentiation)),
                           cf.run_plasticity(C_range_potentiation, start_weight, protocol, param_dict[protocol],
                                             dif_func_list),
                           "r")
        subgraph[loc].plot(range(len(C_range_potentiation)),
                           cf.run_plasticity(C_range_depression, start_weight, protocol,
                                             param_dict[protocol], dif_func_list),
                           "b")
        subgraph[loc].set_xticks([0, duration])
        subgraph[loc].set_xticklabels(['stim' + '\n' + 'start', 'stim' + '\n' + 'end'], fontsize=8)
        # subgraph[loc].set_ylabel('w')
        subgraph[loc].set_xlabel('Time')
    else:
        subgraph[loc].plot(range(len(C_range_potentiation)), cf.run_plasticity(C_range_potentiation, fp_min, protocol,
                                                                      param_dict[protocol], dif_func_list),
                "r", linestyle="--")
        subgraph[loc].plot(range(len(C_range_potentiation)),
                cf.run_plasticity(C_range_potentiation, start_weight, protocol, param_dict[protocol], dif_func_list),
                "r", )
        subgraph[loc].plot(range(len(C_range_potentiation)), cf.run_plasticity(C_range_depression, fp_max, protocol,
                                                                    param_dict[protocol], dif_func_list),
                "b", linestyle="--")
        subgraph[loc].plot(range(len(C_range_potentiation)), cf.run_plasticity(C_range_depression, start_weight, protocol,
                                                                    param_dict[protocol], dif_func_list),
                "b")
        subgraph[loc].set_xticks([0,duration])
        subgraph[loc].set_xticklabels(['stim' + '\n' + 'start','stim' + '\n' + 'end'], fontsize = 8)
        #subgraph[loc].set_ylabel('w')
        subgraph[loc].set_xlabel('Time')
