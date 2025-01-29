import numpy as np
import matplotlib as mpl
from Old import functions_from_Demystifying_Calcium as fdc


def delta_w_matrix(func_list, Calcium_Range, Weight_Range, param_dict):
    deltaW_matrix_list = []
    for i in range(len(func_list)):
        deltaW_matrix = [[func_list[i](c, w, param_dict[i]) for c in Calcium_Range] for w in Weight_Range]
        deltaW_matrix_list.append(deltaW_matrix)
    return deltaW_matrix_list


def delta_w_graph(subgraph, loc, x_axis, lines_for_plot, theta_d, theta_p):
    range_of_colors = np.linspace(0, 1, len(lines_for_plot))
    colors = [mpl.cm.Wistia(x) for x in range_of_colors]
    for n, color in enumerate(colors):
        subgraph[loc].plot(x_axis, lines_for_plot[n], color=color)
    subgraph[loc].set_xticks([theta_d, theta_p])
    subgraph[loc].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
    subgraph[loc].set_xlabel('$\mathregular{[Ca^{2+}]}$', labelpad = -4, fontsize = 13)

def delta_w_matrix(func_list, Calcium_Range, Weight_Range, param_dict):
    deltaW_matrix_list = []
    for i in range(len(func_list)):
        deltaW_matrix = [[func_list[i](c, w, param_dict[i]) for c in Calcium_Range] for w in Weight_Range]
        deltaW_matrix_list.append(deltaW_matrix)
    return deltaW_matrix_list

def delta_w_imshow(subgraph, loc, matrix_for_plot, C_range_in_matrix, weight_range_in_matrix, theta_d, theta_p, norm = None):
    bound = max([abs(np.amin(matrix_for_plot)), abs(np.amax(matrix_for_plot))])
    if norm == None:
        subgraph[loc].imshow(matrix_for_plot, cmap="coolwarm", aspect="auto",
                  extent=[0, C_range_in_matrix[-1], weight_range_in_matrix[0], weight_range_in_matrix[-1]],
                  origin="lower", vmin=-bound, vmax=bound)
        subgraph[loc].set_xticks([theta_d, theta_p])
        subgraph[loc].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
        subgraph[loc].set_xlabel('$\mathregular{[Ca^{2+}]}$', labelpad = -4, fontsize =13 )
    else:
        subgraph[loc].imshow(matrix_for_plot, cmap="coolwarm", aspect="auto",
                                  extent=[0, C_range_in_matrix[-1], weight_range_in_matrix[0],
                                          weight_range_in_matrix[-1]],
                                  origin="lower", norm=norm)
        subgraph[loc].set_xticks([theta_d, theta_p])
        subgraph[loc].set_xticklabels([r'$\theta_D$', r'$\theta_P$'])
        subgraph[loc].set_xlabel('$\mathregular{[Ca^{2+}]}$', labelpad = -4, fontsize =13)


def w_graph_different_fps(subgraph, loc, C_range_potentiation, C_range_depression, start_weight, fp_min, fp_max, protocol, duration,
                           param_dict, dif_func_list=False, no_row_col = False, no_fp = 0):
    if no_row_col == True:
        subgraph.plot(range(len(C_range_potentiation)),
                                fdc.run_plasticity(C_range_potentiation, fp_min, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "r", linestyle=":")
        subgraph.plot(range(len(C_range_potentiation)),
                                fdc.run_plasticity(C_range_potentiation, start_weight, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "r")
        subgraph.plot(range(len(C_range_potentiation)),
                                fdc.run_plasticity(C_range_depression, fp_max, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "b", linestyle=":")
        subgraph.plot(range(len(C_range_potentiation)),
                                fdc.run_plasticity(C_range_depression, start_weight, protocol,
                                                  param_dict[protocol], dif_func_list),
                                "b")
        subgraph.set_xticks([0,duration])
        subgraph.set_xticklabels(['stim' + '\n' + 'start','stim' + '\n' + 'end'], fontsize =13)
        # subgraph[row,col].set_ylabel('w')
        subgraph.set_xlabel('Time (hours)', labelpad = -4, fontsize =13)
        #subgraph.legend(fontsize=9)
    if no_fp:
        subgraph[loc].plot(range(len(C_range_potentiation)),
                           fdc.run_plasticity(C_range_potentiation, start_weight, protocol, param_dict[protocol],
                                             dif_func_list),
                           "r")
        subgraph[loc].plot(range(len(C_range_potentiation)),
                           fdc.run_plasticity(C_range_depression, start_weight, protocol,
                                             param_dict[protocol], dif_func_list),
                           "b")
        subgraph[loc].set_xticks([0, duration])
        subgraph[loc].set_xticklabels(['stim' + '\n' + 'start', 'stim' + '\n' + 'end'], fontsize=13)
        # subgraph[loc].set_ylabel('w')
        subgraph[loc].set_xlabel('Time', labelpad = -4, fontsize = 13)
    else:
        subgraph[loc].plot(range(len(C_range_potentiation)), fdc.run_plasticity(C_range_potentiation, fp_min, protocol,
                                                                      param_dict[protocol], dif_func_list),
                "r", linestyle="--")
        subgraph[loc].plot(range(len(C_range_potentiation)),
                fdc.run_plasticity(C_range_potentiation, start_weight, protocol, param_dict[protocol], dif_func_list),
                "r", )
        subgraph[loc].plot(range(len(C_range_potentiation)), fdc.run_plasticity(C_range_depression, fp_max, protocol,
                                                                    param_dict[protocol], dif_func_list),
                "b", linestyle="--")
        subgraph[loc].plot(range(len(C_range_potentiation)), fdc.run_plasticity(C_range_depression, start_weight, protocol,
                                                                    param_dict[protocol], dif_func_list),
                "b")
        subgraph[loc].set_xticks([0,duration])
        subgraph[loc].set_xticklabels(['stim' + '\n' + 'start','stim' + '\n' + 'end'], fontsize =13)
        #subgraph[loc].set_ylabel('w')
        subgraph[loc].set_xlabel('Time', labelpad = -4,fontsize =13)
