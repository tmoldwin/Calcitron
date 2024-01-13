import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
import constants
from plasticity_rule import Plasticity_Rule
from plasticity_rule import Region
from calcitron import Calcitron
import supervisors as sprv
import pattern_generators as pg
import seaborn as sns
import optuna

TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=TINY_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

class Perceptron_Calcitron(ClassifierMixin):
    def __init__(self, experiment, n_iter=50):
        self.experiment = experiment
        self.classifier_name = experiment.classifier_type
        eta_bias = experiment.eta_bias
        self.n_iter = n_iter
        rule_type = experiment.rule_type
        eta_P = experiment.eta_P
        eta_D = experiment.eta_D
        fP = experiment.fP
        fD = experiment.fD
        if self.classifier_name == 'critic':
            self.regions = [Region(name='N', bounds=(-np.inf, 0.6), fp=0.25, eta=0, fp_color='k', bar_code_color='w'),
                            Region(name='D', bounds=(0.6, 0.9), fp=fD, eta=eta_D, fp_color='b', bar_code_color='b'),
                            Region(name='P', bounds=(0.9, np.inf), fp=fP, eta=eta_P, fp_color='r', bar_code_color='r')]
            self.coeffs = [0.45, 0, 0, 1]
            self.supervisor = sprv.critic_perceptron_supervisor(eta_b=eta_bias)
        elif self.classifier_name == 'target':
            self.regions = Plasticity_Rule(
                [Region(name='N', bounds=(-np.inf, 0.5), fp=0.25, eta=0, fp_color='k', bar_code_color='w'),
                 Region(name='D', bounds=(0.5, 0.6), fp=fD, eta=eta_D, fp_color='b', bar_code_color='b'),
                 Region(name='P', bounds=(0.6, 0.8), fp=fP, eta=eta_P, fp_color='r', bar_code_color='r'),
                 Region(name='PPNZ', bounds=(0.8, np.inf), fp=0.25, eta=0, fp_color='g', bar_code_color='g')])
            self.coeffs = [0.45, 0, 0.1, 1]
            self.supervisor = sprv.target_perceptron_supervisor(eta_b=eta_bias)
        self.plasticity_rule = Plasticity_Rule(self.regions, rule=rule_type)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        self.P = X.shape[0]
        self.N = X.shape[1]
        self.supervisor.make_label_map(X, y)
        self.supervisor.make_label_map(X, y)
        self.label_map = self.supervisor.label_map
        sparsity = np.mean(np.mean(X, axis=1))
        w_init = self.experiment.w_init*np.ones(self.N)
        bias = - np.sum(w_init) * sparsity
        self.model = Calcitron(self.coeffs, self.plasticity_rule, supervisor=self.supervisor, bias=bias)
        SGD_inputs = np.vstack(np.array([X[np.random.permutation(range(self.P))] for i in range(self.n_iter)]))
        self.model.train(SGD_inputs, w_init=w_init)

    def learning_curve(self, plot=True, things_to_plot=['positive_rate', 'negative_rate', 'accuracies']):
        error_history = np.array(self.model.error_history).reshape((self.n_iter, self.P))
        FPs = np.empty(self.n_iter)
        TPs = np.empty(self.n_iter)
        FNs = np.empty(self.n_iter)
        TNs = np.empty(self.n_iter)
        accuracies = np.empty(self.n_iter)
        for i in range(self.n_iter):
            FPs[i] = np.sum(error_history[i] == 'FP')
            TPs[i] = np.sum(error_history[i] == 'TP')
            FNs[i] = np.sum(error_history[i] == 'FN')
            TNs[i] = np.sum(error_history[i] == 'TN')
        accuracies = (TPs + TNs) / (TPs + TNs + FPs + FNs)
        total_positives = TPs + FPs
        total_negatives = FNs + TNs
        FPs = FPs / total_positives
        TPs = TPs / total_positives
        TNs = TNs / total_negatives
        FNs = FNs / total_negatives
        plot_dict = {'FP': FPs, "TPs": TPs, "FNs": FNs, 'TNs': TNs, 'accuracies': accuracies,
                     'positive_rate': total_positives / self.P, 'negative_rate': total_negatives / self.P}
        if plot:
            plt.figure('Learning Curve')
            for key in things_to_plot:
                plt.plot(plot_dict[key], label=key)
            plt.legend()
        return FPs, TPs, FNs, TNs, total_positives, total_negatives, accuracies

    def predict(self, X):
        # Use the trained model to make predictions on new data
        spikes = [self.model.calculate_output(X[i])[1] for i in range(len(X))]
        return spikes

    def score(self, X, y):
        # Calculate the accuracy of the classifier on new data
        return np.mean(np.array(self.predict(X)) == np.array(y))


def run_multiple_trials(experiment, P, N, sparsity, n_iter, n_trials=10):
    # Run multiple trials of the perceptron calcitron
    # and return the learning curves
    FPs = np.empty((n_trials, n_iter))
    TPs = np.empty((n_trials, n_iter))
    FNs = np.empty((n_trials, n_iter))
    TNs = np.empty((n_trials, n_iter))
    accs = np.empty((n_trials, n_iter))
    for i in range(n_trials):
        ('Trial: ', i)
        X, y = pg.generate_perceptron_patterns(P, N, sparsity)
        model = Perceptron_Calcitron(experiment, n_iter=n_iter)
        model.fit(X, y)
        FPs[i], TPs[i], FNs[i], TNs[i], pos, neg, accs[i] = model.learning_curve(plot=False)
    FPs = np.mean(FPs, axis=0)
    TPs = np.mean(TPs, axis=0)
    FNs = np.mean(FNs, axis=0)
    TNs = np.mean(TNs, axis=0)
    acc_mean = np.mean(accs, axis=0)
    acc_STD = np.std(accs, axis=0)
    # plt.plot(acc_mean)
    ('negatives', neg[-1])
    ('pos', pos[-1])
    return FPs, TPs, FNs, TNs, pos, neg, acc_mean, acc_STD


def run_experiments(experiments,
                    P_values=[50, 100, 200], sparsity_values=[0.5], N=100, eta=0.1,
                    n_iter=100, n_trials=2, *args,
                    **kwargs):
    # Create an empty dictionary to store the results
    results_df = pd.DataFrame(columns=['rule_name', 'P', 'sparsity', 'accuracy', 'pos','neg'])
    accuracy_list_df = pd.DataFrame(columns=['rule_name', 'P', 'sparsity', 'iter', 'accuracy', 'accuracy_STD'])
    # Loop through the rule names
    for sparsity in sparsity_values:
        for P in P_values:
            for experiment in experiments:
                # Run the run_multiple_trials() function with the current values
                FPs, TPs, FNs, TNs, pos, neg, acc_mean, acc_std = run_multiple_trials(experiment, P, N, sparsity,
                                                                            n_iter, n_trials)
                row_df = pd.DataFrame({'rule_name': experiment.rule_name, 'P': P, 'sparsity': sparsity,
                                       'accuracy': acc_mean[-1], 'pos':pos[-1], 'neg':neg[-1]}, index=[0])
                results_df = pd.concat((results_df, row_df), ignore_index=True)
                acc_row = pd.DataFrame({'rule_name': experiment.rule_name, 'P': P, 'sparsity': sparsity,
                                        'iter': range(n_iter), 'accuracy': acc_mean,
                                        'accuracy_STD': acc_std})
                accuracy_list_df = pd.concat((accuracy_list_df, acc_row), ignore_index=True)
                (results_df.tail(1))
                (accuracy_list_df.tail(1))
        # Open a JSON file for writing
                results_df.to_csv(constants.DATA_FOLDER + "perceptron_results.csv")
                accuracy_list_df.to_csv(constants.DATA_FOLDER + "perceptron_accuracy_list.csv")
                # dashboard()
                # plt.pause(20)

def dashboard():
    plt.close('all')
#    fig, ax = plt.subplots(nrows=4, ncols=4)
    results_df = pd.read_csv(constants.DATA_FOLDER + "perceptron_results.csv")
    accuracy_list_df = pd.read_csv(constants.DATA_FOLDER + "perceptron_accuracy_list.csv")
    # Plot the lines on two facets
    sns.relplot(
        data=accuracy_list_df, x="iter", y="accuracy",
        hue="rule_name", col="sparsity", row="P",
        kind="line", facet_kws=dict(sharex=True, sharey=True)
    )
    plt.tight_layout()
    sns.catplot(x="P", y="accuracy", hue="rule_name", col="sparsity",
                data=results_df, kind="bar",
                height=4, aspect=.7)


class Exp_Params:
    def __init__(self, rule_name, rule_type, fd, fp, eta = 0.0005, classifier_type = 'critic'):
        self.classifier_type = 'critic'
        self.rule_name = rule_name
        self.rule_type = rule_type
        self.fD = fd
        self.fP = fp
        self.set_etas(eta)
        self.w_init = (self.fD+self.fP)/2
        if np.isnan(self.w_init):
            self.w_init = 0.5
        (self.w_init)
        self.classifier_type = 'critic'

    def set_etas(self, eta):
        self.eta_P = eta
        self.eta_bias = eta
        self.eta_D = eta
        if self.rule_type == 'linear':
            self.eta_D = -eta

    def __str__(self):
        return self.rule_name

if __name__ == "__main__":
    experiments = [Exp_Params("linear", "linear", -np.nan, np.nan, eta=0.00139),
                   Exp_Params("linear LB", "linear", 0, np.nan, eta= 0.0004167644213458093),
                   Exp_Params("FPLR 10", "linear", 0, 10, eta=    0.05440695056830015),
                   Exp_Params("linear LUB 10", "linear", 0, 10, eta=0.0009737143080057794),
                   Exp_Params("FPLR 1", "FPLR", 0, 1, eta=0.02660214576449791),  # optimized for p = 100 sparsity = 500
                   Exp_Params("linear LUB 1", "linear", 0, 1, eta= 0.0004202233589429778)]
    #                ]
    # 0.073513883
    # 0.00311668
    # 0.011119419
    # 0.076237783
    # 0.000227045
    # 0.005671719

    0.0013916587197212963
    0.0004167644213458093
    0.05440695056830015
    0.0009737143080057794
    0.02660214576449791
    0.0004202233589429778

    #experiments = [Exp_Params("FPLR 5", "FPLR", 0, 5, eta=0.0000115)]
    # run_experiments(experiments, n_trials=10, P_values=[25, 50,100], sparsity_values=[0.5],
    #                 n_iter=1000)
    dashboard()


