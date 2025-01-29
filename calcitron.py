import collections
import numpy as np
import matplotlib as mpl
import six
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import pandas as pd
import supervisors
import supervisors as sprv
import plot_helpers as ph
import param_helpers as pah
#plt.rc('xtick', labelsize=6)  # fontsize of the tick labels

cmap = mpl.cm.get_cmap('Wistia')
class Calcitron:
    def __init__(self, coeffs, plasticity_rule, supervisor=sprv.null_supervisor(),
                 activation_function = 'threshold', bias = 0, w_init = 'middle'):
        #possible activation functions are 'linear', 'threshold'
        self.coeffs = coeffs
        self.alpha = coeffs[0]
        self.beta = coeffs[1]
        self.gamma = coeffs[2]
        self.delta = coeffs[3]
        self.name = "base"
        self.plasticity_rule = plasticity_rule
        self.activation_function = activation_function
        self.bias = bias
        self.supervisor = supervisor
        self.w_init = w_init
        self.N = None

    def to_pandas(self):
        # Use the to_pandas function from paramfunction file for coeffs
        df_coeffs = pah.coeffs_to_pandas(self.coeffs)
        # Get the plasticity rule DataFrame
        df_plasticity_rule = self.plasticity_rule.to_pandas()
        # Get the supervisor DataFrame
        df_supervisor = self.supervisor.to_pandas()
        # Concatenate all DataFrames
        df = pd.concat([df_coeffs, df_plasticity_rule, df_supervisor], axis=1)
        df['b'] = round(self.bias,2)
        df['g'] = self.activation_function
        df['N'] = self.N
        # Add the activation function
        # Convert column names to Unicode
        df.columns = pah.latex_to_unicode(df.columns)
        return df

    def __str__(self):
        return (f"Plasticity rule: {str(self.plasticity_rule)}\n"
                f"Coefficients: {self.coeffs}\n")

    def calculate_output(self, x): #if no supervising signal is defined output is just w dot x
        weighted_sum = np.dot(self.weights, np.array(x))
        if self.activation_function == 'threshold':
            output = int((weighted_sum + self.bias) > 0)
        elif self.activation_function == 'linear':
            output = weighted_sum
        return weighted_sum, output

    def calculate_calcium(self, x, Z, output):
        C_local = self.alpha * np.array(x)
        C_tff = self.beta * np.dot(self.weights, np.array(x))
        C_bap = self.gamma * output
        C_sprv = self.delta * Z
        C_global = C_tff + C_bap + C_sprv
        C_total = C_local + C_global
        return C_local, C_tff, C_bap, C_sprv, C_global, C_total

    def input_imshow(self, ax = None, plot_cbar = 1):
        input = self.X.T
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(input, aspect = 'auto', cmap = ListedColormap(['white', 'black']), origin = 'lower')
        ax.set_ylabel(r'$\bf{Input}$' + '\nsyn #')
        ax.set_xticks([1, input.shape[1] / 2, input.shape[1]])
        ax.set_yticks([0, self.N-1], labels = [1, self.N])
        if plot_cbar != 0:
            cbar = plt.colorbar(im)
            cbar.set_ticks([0.25,0.75], labels = [0,1])
            cbar.set_label('$\mathregular{x}$')

    def bar_code_imshow(self, ax = None, plot_cbar = 1):
        Ca = self.C_tot.T
        if ax is None:
            fig, ax = plt.subplots()
        bar_codes = self.plasticity_rule.bar_code_from_C(Ca)
        im = self.plasticity_rule.bar_code_imshow(Ca, ax = ax, plot_cbar = plot_cbar)
        ax.set_xticks([1, Ca.shape[1] / 2, Ca.shape[1]])
        ax.set_yticks([0, self.N-1], labels = [1, self.N])
        return im

    def Ca_total_imshow(self, ax = None, plot_cbar = 1, C_range = (None,None)):
        Ca = self.C_tot.T
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(Ca, aspect = 'auto', cmap='Wistia', origin='lower', vmax = C_range[1], vmin = C_range[0])
        ax.set_ylabel(r'$\bf{C_{total}^{i}}$' + '\nsyn #')
        if plot_cbar != 0:
            cbar = plt.colorbar(im)
            cbar.set_label(r'$\mathregular{C_{Total}^{i}}$')
        ax.set_yticks([0, self.N-1], labels = [1, self.N])
        return im

    def weight_imshow(self, ax = None, plot_cbar = 1):
        weights = self.weight_history.T
        if ax is None:
            fig, ax = plt.subplots()
        min_weight = min(self.plasticity_rule.fps)
        max_weight = max(self.plasticity_rule.fps)
        im = ax.imshow(weights, aspect='auto', cmap = 'Reds',
                       origin = 'lower', vmin=min_weight, vmax=max_weight)
        ax.set_xticks([1, weights.shape[1] / 2, weights.shape[1]])
        ax.set_yticks([0, self.N-1], labels = [1, self.N])
        ax.set_ylabel(r'$\bf{Weights}$' +'\nsyn #')
        if plot_cbar != 0:
            cbar = plt.colorbar(im)
            cbar.set_label('$\mathregular{w}$')
        return im

    def output_stem(self, ax = None):
        V = self.y_hat
        if ax is None:
            fig, ax = plt.subplots()
        label_frmt_dct = {'TP': 'ro', 'TN': 'ko', 'FP': 'rx', 'FN': 'kx',
                          'H': 'rx', 'M': 'go', 'L': 'bx'}
        x = np.arange(len(V))
        markersz = 4
        label_values = set(self.error_history)
        for label in label_values:
            label_inds = indices = [i for i, item in enumerate(self.error_history) if item == label]
            label_Vs = V[label_inds]
            markerline, stemline, baseline = ax.stem(x[label_inds], label_Vs, linefmt="k",
                                                     markerfmt=label_frmt_dct[label], basefmt="k")
            plt.setp(markerline, markersize=markersz)
        if self.activation_function == 'threshold':
            ax.plot(-self.bias_history, color="k", linestyle=":")
        if isinstance(self.supervisor,supervisors.homeostatic_supervisorPD):
            ax.axhline(self.supervisor.min_target, color="k", linestyle=":")
            ax.axhline(self.supervisor.max_target, color="k", linestyle=":")
        ax.set_ylim(0.8 * min(V), 1.5 * max(V))
        ax.set_ylabel(r'$\bf{Output}$' + '\n$\sum_{i} w_ix_i$')

    def Z_stem(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        Z = self.Z
        x = np.arange(len(Z))
        ax.stem(x, Z, linefmt="k", markerfmt="", basefmt="k")
        ax.set_ylabel(r'$\bf{Supervisor}$' + '\nZ')

    def all_plots(self, axes = None, plot_cbar = 1, C_range = (None,None)):
        what_to_plot = axes.keys()
        ax_list = list(axes.values())
        if 'input' in what_to_plot:
            self.input_imshow(axes['input'], plot_cbar = plot_cbar)
        if 'output' in what_to_plot:
            self.output_stem(axes['output'])
        if 'Z' in what_to_plot:
            self.Z_stem(axes['Z'])
        if 'C' in what_to_plot:
            self.Ca_total_imshow(axes['C'], plot_cbar = plot_cbar, C_range = C_range)
        if 'bar_code' in what_to_plot:
            self.bar_code_imshow(ax = axes['bar_code'], plot_cbar = plot_cbar)
        if 'w' in what_to_plot:
            self.weight_imshow(ax = axes['w'], plot_cbar = plot_cbar)
        ax_list[-1].set_xlabel('Timestep')
        ph.share_axes(list(axes.values()), sharex = 'both', sharey=0)

    def initialize_weights(self, w_init ='middle', sparsity = None, seed = 42):
        min_weight = min(self.plasticity_rule.fps)
        max_weight = max(self.plasticity_rule.fps)
        if sparsity is None:
            sparsity = self.N
        if np.isnan(min_weight):
            min_weight = 0
        if np.isnan(max_weight):
            max_weight = 1
        if isinstance(w_init, (collections.abc.Sequence, np.ndarray))\
                and not isinstance(w_init, six.string_types): #initializes weights to predefined values
            return w_init
        elif w_init == 'middle': #initializes all weights to middle fixed point
            return np.ones(self.N) * (min_weight + 0.5 * (max_weight-min_weight))
        elif w_init == 'random':
            return min_weight + np.random.random(self.N) * (max_weight - min_weight) #initializes weights to uniform betwee max and min weight
        elif isinstance(w_init, (float, int)):
            return np.ones(self.N) * w_init
    def train(self, X, momentum = False):
        X = np.squeeze(np.atleast_2d(X))
        P = X.shape[0]
        self.momentum = momentum
        self.N = X.shape[1]
        history_size = (P, self.N)
        self.X = X
        self.y_hat_binary = np.empty(P)
        self.y_hat = np.empty(P)
        self.C_tot = np.empty(history_size)
        self.delta_w = np.empty(history_size)
        self.C_loc = np.empty(history_size)
        self.C_TFF = np.empty(history_size)
        self.C_BAP = np.empty(history_size)
        self.C_SPRV = np.empty(history_size)
        self.C_glob = np.empty(P)
        self.weights = self.initialize_weights(w_init = self.w_init)
        self.weight_history = np.zeros(history_size)
        self.Z = np.empty(P)
        self.error_history = []
        self.bias_history = np.empty(P)

        for i in range(P):
            self.weight_history[i,:] = self.weights
            self.y_hat_binary[i] = self.calculate_output(X[i])[1]
            self.y_hat[i] = self.calculate_output(X[i])[0]
            self.Z[i], error = self.supervisor.supervise(i, X[i], self.y_hat_binary[i], self.y_hat[i])
            self.error_history.append(error)
            C_local, C_tff, C_bap, C_sprv, C_global, C_total = self.calculate_calcium(X[i], self.Z[i], self.y_hat_binary[i])
            self.C_loc[i,:] = C_local
            self.C_TFF[i] = C_tff
            self.C_BAP[i] = C_bap
            self.C_SPRV[i] = C_sprv
            self.C_glob[i] = C_global
            self.C_tot[i,:] = C_total
            self.weights = self.plasticity_rule.update_weights(C_total, self.weights)
            self.bias += self.supervisor.delta_bias(i, X[i], self.y_hat_binary[i], self.y_hat[i])
            self.bias_history[i] = self.bias
        return X, self.C_tot, self.y_hat, self.weight_history


if __name__ == "__main__":
    pass





