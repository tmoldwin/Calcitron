import constraint
import numpy as np

from matplotlib import pyplot as plt

problem = constraint.Problem()

problem.addVariable('alpha', np.arange(0.1,1,0.1))
problem.addVariable('gamma', [0])
problem.addVariable('Z_d', np.arange(0.1,1,0.1))
problem.addVariable('Z_p', np.arange(0.1,1,0.1))
problem.addVariable('theta_p', np.arange(0.1,1,0.1))
problem.addVariable('theta_d', np.arange(0.1,1,0.1))
problem.addVariable('F_p', [0.9])
problem.addVariable('F_d', [0.7])

# def our_constraint1(alpha, gamma, Z_d, Z_p, theta_p, theta_d, F_p, F_d):
#     if F_p>F_d and theta_p>theta_d \
#             and alpha * F_p < theta_d and gamma<(alpha * F_p + gamma) < theta_d and Z_d<Z_p<theta_d\
#             and theta_d < (alpha * F_d + Z_d) < (alpha * F_p + Z_d) < (alpha * F_d + gamma + Z_d) < (alpha * F_p + gamma + Z_d) < theta_p \
#             and theta_p < (alpha * F_d + Z_p) < (alpha * F_p + Z_p):
#         return True

def our_constraint2(alpha, gamma, Z_d, Z_p, theta_p, theta_d, F_p, F_d):
    if F_p>F_d and theta_p>theta_d \
            and alpha * F_p < theta_d and Z_d<Z_p<theta_d\
            and theta_d < (alpha * F_d + Z_d) < (alpha * F_p + Z_d) < theta_p \
            and theta_p < (alpha * F_d + Z_p) < (alpha * F_p + Z_p):
        return True


problem.addConstraint(our_constraint2, ['alpha', 'gamma', 'Z_d', 'Z_p', 'theta_p', 'theta_d', 'F_p', 'F_d'])
solutions = problem.getSolutions()
# iter = problem.getSolutionIter()
# (next(iter))

#Easier way to  and see all solutions
for solution in solutions:
    (solution)

#Prettier way to  and see all solutions
# length = len(solutions)
# (length)
# ("('alpha','gamma' 'delta', 'theta_ppnz' , 'theta_p', 'theta_d', 'F_p', 'F_d') ∈ {", end="")
# for index, solution in enumerate(solutions):
#     if index == length - 1:
#         ("({},{},{},{},{})".format(round(solution['alpha'], 1),round(solution['delta'],1), round(solution['theta_d'],1)
#                                      , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
#     else:
#         ("({},{},{},{},{}),".format(round(solution['alpha'],1), round(solution['delta'],1), round(solution['theta_d'],1)
#                                       , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
# ("}")


# param_dict_bar2 = {'theta_d': 0.5, 'theta_p': 0.6, 'F_d': 0.7000000000000001, 'F_p': 0.9, 'Z_d': 0.2, 'Z_p': 0.4, 'alpha': 0.3, 'gamma': 0.1}
param_dict_bar2 = {'F_d': 0.7, 'F_p': 0.9, 'gamma': 0, 'Z_d': 0.2, 'Z_p': 0.5, 'alpha': 0.6, 'theta_d': 0.6, 'theta_p': 0.9}
# x=['alpha', 'gamma', 'Z_d', 'Z_p', 'F_p', 'F_d', 'theta_d', 'theta_p']
# y = [0.3, 0.1, 0.2, 0.4, 0.9, 0.7, 0.5, 0.6]
x_bar = ["local", "BAP", "false-positive \n supervisor","false-negative \n supervisor", "local+BAP", "local+BAP+false- \n positive supervisor", "local+false- \n negative supervisor"]
alpha_F_d_vector = np.array([1,0,0,0,1,1,1])
alpha_F_p_vector = np.array([1,0,0,0,1,1,1])
gamma_vector = np.array([0,1,0,0,1,1,0])
Zd_vector = np.array([0,0,1,0,0,1,0])
Zp_vector = np.array([0,0,0,1,0,0,1])
y_Fd = param_dict_bar2['alpha']*param_dict_bar2['F_d']*alpha_F_d_vector
y_Fp = param_dict_bar2['alpha']*param_dict_bar2['F_p']*alpha_F_p_vector
delta_FP_FD = y_Fp - y_Fd
y_gamma = param_dict_bar2['gamma']*gamma_vector
y_Zd = param_dict_bar2['Z_d']*Zd_vector
y_Zp = param_dict_bar2['Z_p']*Zp_vector

plt.figure(1)
plt.bar(x_bar,y_Fd, color = "tab:orange", label = r'$\alpha$' + "*F_d")
plt.bar(x_bar,delta_FP_FD , bottom= y_Fd,color = "r", label = r'$\alpha$' + "*F_p")
plt.bar(x_bar,y_gamma, bottom=y_Fd+delta_FP_FD, color = "b", label = r'$\gamma$')
plt.bar(x_bar,y_Zd, bottom = y_Fd+delta_FP_FD+y_gamma, color = "g", label = "Z_d")
plt.bar(x_bar,y_Zp, bottom = y_Fd+delta_FP_FD+y_gamma, color = "limegreen", label = "Z_p")
plt.axhline(param_dict_bar2['theta_d'], color = "k", linestyle = ":")
plt.axhline(param_dict_bar2['theta_p'], color = "k", linestyle = ":")
plt.xticks(rotation=15)
plt.legend()
#
#
plt.figure(2)
plt.bar(x_bar,y_Fd, color = "tab:orange", label = r'$\alpha$' + "*F_d")
plt.bar(x_bar,y_gamma, bottom=y_Fd, color = "b", label = r'$\gamma$')
plt.bar(x_bar,y_Zd, bottom = y_Fd+y_gamma, color = "g", label = "Z_d")
plt.bar(x_bar,y_Zp, bottom = y_Fd+y_gamma, color = "limegreen", label = "Z_p")
plt.axhline(param_dict_bar2['theta_d'], color = "k", linestyle = ":")
plt.axhline(param_dict_bar2['theta_p'], color = "k", linestyle = ":")
plt.xticks(rotation=15)
plt.legend()
plt.show()



