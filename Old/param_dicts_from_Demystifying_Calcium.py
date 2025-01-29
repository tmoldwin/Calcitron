import numpy as np
theta_d = 0.45
theta_p = 0.85


# eta_dict_no_drift_SBC = {(-np.inf, theta_d): 0, (theta_d, theta_p): 0.15,
#                      (theta_p, np.inf): 0.25}
eta_dict_no_drift_SBC = {(-np.inf, theta_d): 0, (theta_d, theta_p): 0.15,
                     (theta_p, np.inf): 0.15}
FP_calcium_dict_SBC = {(0, theta_d): 0.5, (theta_d, theta_p): 0.25,
                                     (theta_p, np.inf): 1}

param_dict_for_plasticity_graphs_sbc = [
    {"eta": 0.2, "hard_threshold": 0, "k_d": 0.25, "k_p": 1, "theta_d": theta_d, "theta_p": theta_p,
     "b1": 80, "b2": 80},
    {"eta_dict": eta_dict_no_drift_SBC, "FP_calcium_dict": FP_calcium_dict_SBC, "soft_threshold": 0}]
