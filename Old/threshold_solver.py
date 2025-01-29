import constraint
import numpy as np
import Calcitron_functions as cf

problem = constraint.Problem()

problem.addVariable('alpha', [0.7])
problem.addVariable('delta', [0.9])
problem.addVariable('theta_p', [cf.theta_p])
problem.addVariable('theta_d',[cf.theta_d])
problem.addVariable('eta',np.arange(0,1.5,0.1))
problem.addVariable('r',np.arange(0.01,9,0.05))


def our_constraint1(alpha, delta, theta_p, theta_d, eta, r, dist = 1):
     if (alpha > theta_p-theta_d) and (alpha < theta_d) and (delta < theta_d) and (alpha+delta > theta_p) \
             and (theta_p>theta_d) and (alpha*(1+eta)<theta_d) and ((alpha*(1+eta)*(np.e**(-dist**2/r))+delta)>theta_d) \
             and ((alpha*(1+eta)*(np.e**(-dist**2/r))+delta)<theta_p) and ((alpha*1*(np.e**(-dist**2/r))+delta)<theta_d):
         return True

# def our_constraint2(alpha, delta, theta_p, theta_d, eta, r, dist = 2):
#     if (alpha*(1+eta)*(np.e**(-dist**2/r))>theta_d) \
#              and (alpha*(1+eta)*(np.e**(-dist**2/r))<theta_p):
#         #and (((alpha*np.e**(-1/r))<theta_d)
# (((alpha*np.e**(-1/r))+delta)<theta_d) or \
#              (((alpha * np.e ** (-1 / r)) + delta) > theta_p)):
#         return True


#problem.addConstraint(our_constraint2, ['alpha', 'delta', 'theta_p', 'theta_d', 'eta', 'r'])
problem.addConstraint(our_constraint1, ['alpha', 'delta', 'theta_p', 'theta_d', 'eta', 'r'])

solutions = problem.getSolutions()


# Easier way to  and see all solutions
for solution in solutions:
    (solution)

# Prettier way to  and see all solutions
length = len(solutions)
(length)
("('alpha', 'delta', 'theta_p', 'theta_d', 'eta', 'r') ∈ {", end="")
# for index, solution in enumerate(solutions):
#     if index == length - 1:
#         ("({},{},{},{},{})".format(round(solution['alpha'], 1),round(solution['delta'],1), round(solution['theta_d'],1)
#                                      , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
#     else:
#         ("({},{},{},{},{}),".format(round(solution['alpha'],1), round(solution['delta'],1), round(solution['theta_d'],1)
#                                       , round(solution['theta_p'],1), round(solution['eta'],1), round(solution['r'], 1)), end="\n")
# ("}")


