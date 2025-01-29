import numpy as np
import matplotlib as mpl

def calculate_plasticity(C, weight, protocol, param_dict, N):
    delta_w = protocol(C, weight, param_dict,N)
    return delta_w

#add soft omega after make it usable all the tme with param dict

def step_function(x, dict):
    (x)
    for interval in dict.keys():
        if x >= interval[0] and x < interval[1]:
            return dict[interval]

def modified_shouval_array(Ca,weights, param_dict, N):
    delta_w = np.zeros(N)
    eta_dict, FP_dict, soft_threshold = \
        param_dict["eta_dict"], param_dict["FP_dict"], param_dict["soft_threshold"]
    for interval in eta_dict.keys():
        is_in_interval = np.bitwise_and((np.heaviside(Ca-interval[0], 0)).astype(int) ,(np.heaviside(interval[1]-Ca, 0).astype(int)))
        is_in_interval_updated = eta_dict[interval]*is_in_interval*(FP_dict[interval] - weights)
        delta_w += is_in_interval_updated
    return delta_w


def modified_shouval_array_linear(Ca,weights, param_dict, N):
    delta_w = np.zeros(N)
    eta_dict, FP_dict, soft_threshold = \
        param_dict["eta_dict"], param_dict["FP_dict"], param_dict["soft_threshold"]
    for interval in eta_dict.keys():
        is_in_interval = np.bitwise_and((np.heaviside(Ca-interval[0], 0)).astype(int) ,(np.heaviside(interval[1]-Ca, 0).astype(int)))
        is_in_interval_updated = eta_dict[interval]*is_in_interval*FP_dict[interval]
        delta_w += is_in_interval_updated
    return delta_w


def modified_shouval_math(Ca, w, param_dict):
    eta_dict, FP_calcium_dict, soft_threshold = \
        param_dict["eta_dict"], param_dict["FP_calcium_dict"], param_dict["soft_threshold"]
    if soft_threshold == 1:
        return soft_omega(Ca) * (soft_omega(Ca, k=[0.5,0, 1], theta= [0.3, 1.1],b=[100,100,100,100])- w)
    else:
        return step_function(Ca, eta_dict) * (step_function(Ca, FP_calcium_dict)- w)


def calculate_momentum_update(raw_delta_vec, m,v,beta1 = 0.9,beta2 = 0.999, eps = 1e-8):
    """
    Momentum calculation - Receives 1xN row of deltas, and instance variables m and v
    Returns same, make sure to multiply by learning rate outside
    """
    m = beta1*m + (1-beta1)*raw_delta_vec
    v = beta2*v + (1-beta2)*(raw_delta_vec**2)
    momentum_delta_vec = np.squeeze(m/(np.sqrt(v)+eps))#deleted a lr, and a minus for locations, + for bias, - for weights
    return momentum_delta_vec, m, v

def add_colorbar_outside(im,ax):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width = 0.01
    eps = 0.01 #margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    return cbar


def delta_w_graph(subgraph, col, x_axis, lines_for_plot, theta_d, theta_p):
    range_of_colors = np.linspace(0, 1, len(lines_for_plot))
    colors = [mpl.cm.Wistia(x) for x in range_of_colors]
    for n, color in enumerate(colors):
        subgraph[col].plot(x_axis, lines_for_plot[n], color=color)
    subgraph[col].set_xticks([theta_d, theta_p])
    subgraph[col].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
    subgraph[col].set_xlabel('$\mathregular{[Ca^{2+}]}$', fontsize = 15)


def delta_w_imshow(subgraph, col, matrix_for_plot, C_range_in_matrix, weight_range_in_matrix, theta_d, theta_p, norm = None):
    bound = max([abs(np.amin(matrix_for_plot)), abs(np.amax(matrix_for_plot))])
    if norm == None:
        subgraph[col].imshow(matrix_for_plot, cmap="coolwarm", aspect="auto",
                  extent=[0, C_range_in_matrix[-1], weight_range_in_matrix[0], weight_range_in_matrix[-1]],
                  origin="lower", vmin=-bound, vmax=bound)
        subgraph[col].set_xticks([theta_d, theta_p])
        subgraph[col].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
        subgraph[col].set_xlabel('$\mathregular{[Ca^{2+}]}$', fontsize = 15)
    else:
        subgraph[col].imshow(matrix_for_plot, cmap="coolwarm", aspect="auto",
                                  extent=[0, C_range_in_matrix[-1], weight_range_in_matrix[0],
                                          weight_range_in_matrix[-1]],
                                  origin="lower", norm=norm)
        subgraph[col].set_xticks([theta_d, theta_p])
        subgraph[col].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
        subgraph[col].set_xlabel('$\mathregular{[Ca^{2+}]}$', fontsize = 15)

def run_plasticity(C, weight_0, protocol, param_dict, dif_func_list):
    weight_over_time = np.zeros(len(C))
    weight_over_time[0] = weight_0
    for i in range(len(C) - 1):
        delta_w = calculate_plasticity(C[i], weight_over_time[i], protocol, param_dict, dif_func_list)
        weight_over_time[i + 1] = weight_over_time[i] + delta_w
    return weight_over_time


def w_graph_one_eta(subgraph, col, C_range_potentiation, C_range_depression, start_weight, protocol, duration,
                           param_dict, N, dif_func_list=False, no_row_col = False):
    subgraph[col].plot(range(len(C_range_potentiation)),[calculate_plasticity(C_i, start_weight, protocol,
                                    param_dict,N ) for C_i in C_range_potentiation] ,"r")
    subgraph[col].plot(range(len(C_range_potentiation)), [calculate_plasticity(C_i, start_weight, protocol,
                                              param_dict, N) for C_i in C_range_depression],"b")
    ("d", [calculate_plasticity(C_i, start_weight, protocol,
                                              param_dict, N) for C_i in C_range_depression])
    ("p", [calculate_plasticity(C_i, start_weight, protocol,
                                    param_dict,N ) for C_i in C_range_potentiation])
    subgraph[col].set_xticks([0,duration])
    subgraph[col].set_xticklabels(['stim start', 'stim end'], fontsize = 15)
    subgraph[col].set_ylabel('w',fontsize = 15)
    subgraph[col].set_xlabel('Time', fontsize = 15)
