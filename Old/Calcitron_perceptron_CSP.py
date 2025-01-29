import constraint
import numpy as np

from matplotlib import pyplot as plt

problem = constraint.Problem()

problem.addVariable('alpha', np.arange(0.1,1,0.1))
problem.addVariable('gamma', np.arange(0.1,1,0.1))
problem.addVariable('delta', np.arange(0.1,1,0.1))
problem.addVariable('theta_ppnz', [0.8])
problem.addVariable('theta_p', [0.6])
problem.addVariable('theta_d', [0.5])
problem.addVariable('F_p', np.arange(0.1,1,0.1))
problem.addVariable('F_d', np.arange(0.1,1,0.1))

def our_constraint1(alpha, gamma, delta,theta_ppnz , theta_p, theta_d, F_p, F_d):
    if F_p>F_d and theta_p>theta_d and theta_ppnz>theta_p \
            and alpha * F_p < theta_d and (gamma + delta) < theta_d \
            and theta_d < (alpha * F_d + gamma) < (alpha * F_p + gamma) < theta_p \
            and theta_p < (alpha * F_d + delta) < (alpha * F_p + delta) < theta_ppnz \
            and theta_ppnz < (alpha * F_d + gamma + delta) < (alpha * F_p + gamma + delta) \
            and 1 == 1:
        return True


problem.addConstraint(our_constraint1, ['alpha', 'gamma', 'delta', 'theta_ppnz', 'theta_p', 'theta_d', 'F_p', 'F_d'])
solutions = problem.getSolutions()


# Easier way to  and see all solutions
for solution in solutions:
    (solution)

# Prettier way to  and see all solutions
length = len(solutions)
(length)
# ("('alpha','gamma' 'delta', 'theta_ppnz' , 'theta_p', 'theta_d', 'F_p', 'F_d') ∈ {", end="")
# for index, solution in enumerate(solutions):
#     if index == length - 1:
#         ("({},{},{},{},{})".format(round(solution['alpha'], 1),round(solution['delta'],1), round(solution['theta_d'],1)
#                                      , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
#     else:
#         ("({},{},{},{},{}),".format(round(solution['alpha'],1), round(solution['delta'],1), round(solution['theta_d'],1)
#                                       , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
# ("}")

param_dict_bar = {'alpha':0.6,'gamma':0.1, 'delta':0.3, 'F_p':0.8, 'F_d':0.7}
x=['alpha', 'gamma', 'delta', 'F_p', 'F_d', 'theta_d', 'theta_p', 'theta_ppnz']
y = [0.6, 0.1, 0.3, 0.8, 0.7, 0.5, 0.6, 0.8]
x_bar = ["local", "BAP", "supervisor", "BAP + supervisor",  "local+BAP", "local+supervisor", "local+BAP+supervisor"]
alpha_F_d_vector = np.array([1,0,0,0,1,1,1])
alpha_F_p_vector = np.array([1,0,0,0,1,1,1])
gamma_vector = np.array([0,1,0,1,1,0,1])
delta_vector = np.array([0,0,1,1,0,1,1])
y_Fd = param_dict_bar['alpha']*param_dict_bar['F_d']*alpha_F_d_vector
y_Fp = param_dict_bar['alpha']*param_dict_bar['F_p']*alpha_F_p_vector
delta_FP_FD = y_Fp - y_Fd
y_gamma = param_dict_bar['gamma']*gamma_vector
y_delta = param_dict_bar['delta']*delta_vector

plt.figure(1)
plt.bar(x_bar,y_Fd, color = "tab:orange", label = r'$\alpha$' + "*F_d")
plt.bar(x_bar,delta_FP_FD , bottom= y_Fd,color = "r",label = r'$\alpha$' + "*F_p")
plt.bar(x_bar,y_gamma, bottom=y_Fd+delta_FP_FD, color = "b",label = r'$\gamma$')
plt.bar(x_bar,y_delta, bottom = y_Fd+delta_FP_FD+y_gamma, color = "g", label = r'$\delta$')
plt.axhline(0.8, color = "k", linestyle = ":")
plt.axhline(0.6, color = "k", linestyle = ":")
plt.axhline(0.5, color = "k", linestyle = ":")
plt.legend()
plt.xticks(rotation=15)


plt.figure(2)
plt.bar(x_bar,y_Fd, color = "tab:orange", label = r'$\alpha$' + "*F_d")
plt.bar(x_bar,y_gamma, bottom=y_Fd, color = "b", label = r'$\gamma$')
plt.bar(x_bar,y_delta, bottom = y_Fd+y_gamma, color = "g", label = r'$\delta$')
plt.axhline(0.8, color = "k", linestyle = ":")
plt.axhline(0.6, color = "k", linestyle = ":")
plt.axhline(0.5, color = "k", linestyle = ":")
plt.xticks(rotation=15)
plt.legend()
plt.show()