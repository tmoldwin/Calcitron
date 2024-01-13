import optuna
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import constants
from percep_perform import Exp_Params, run_multiple_trials


def objective(trial, experiment, P, N, sparsity, n_iter, n_trials):
    eta = trial.suggest_float('eta', 1e-6, 1e-1, log=True)
    # experiment = Exp_Params("linear LUB 5", "linear", 0, 5, eta=eta)
    experiment.set_etas(eta)
    FPs, TPs, FNs, TNs, pos, neg, acc_mean, acc_std = run_multiple_trials(experiment, P, N, sparsity, n_iter, n_trials)
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    plt.figure(experiment.rule_name)
    plt.plot(acc_mean, label='eta = ' + str(np.round(eta,5)))
    plt.legend()
    plt.show()
    plt.pause(15)
    error = 1 - acc_mean[-1]
    ('error', error)
    return error  # An objective value linked with the Trial object.

experiments = [Exp_Params("linear", "linear", -np.nan, np.nan, eta=0.01),
               Exp_Params("linear LB", "linear", 0, np.nan, eta=0.01),
               Exp_Params("FPLR 10", "linear", 0, 10, eta=0.01),
               Exp_Params("linear LUB 10", "linear", 0, 10, eta=0.001),
               Exp_Params("FPLR 1", "FPLR", 0, 1, eta=0.025),  # optimized for p = 100 sparsity = 500
               Exp_Params("linear LUB 1", "linear", 0, 1, eta=0.001)
               ]

# experiments = [Exp_Params("linear LUB 5", "linear", 0, 5), Exp_Params("FPLR 5", "FPLR", 0, 5, eta=0.025)]
best_params_DF = pd.DataFrame(columns=['rule_name', 'eta', 'acc_mean'])
for experiment in experiments:
    obj_func = lambda trial: objective(trial, experiment, P=150, N=100, sparsity=0.5, n_iter=1000, n_trials=5)
    # Pass func to Optuna studies
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_func, n_trials=10)
    best_params = study.best_params
    best_params_DF = best_params_DF.append({'rule_name':experiment.rule_name,
                                            'eta':best_params['eta'],
                                            'score':study.best_value} ,ignore_index=True)
    #study.trials_dataframe().to_csv(constants.DATA_FOLDER + experiment.rule_name + '.csv')
    best_params_DF.to_csv(constants.DATA_FOLDER + 'best_params.csv')
