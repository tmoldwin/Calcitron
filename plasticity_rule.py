import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm, ListedColormap
import constants
import pandas as pd
import param_helpers as ph

canonical_delay_dur = 5
canonical_stim_dur = 30
canonical_end_dur = 30

default_fp_colors = {"P":'r', "D":'b', 'N':'k'}
default_bc_colors = {"P": 'r', "D": 'b', 'N': 'w'}

def step_function(x, thresholds, heights, theta_0=0):
    thetas = [theta_0] + thresholds  # append the zero threshold
    sort_inds = np.argsort(thetas)
    sorted_thresholds = np.array(thetas)[sort_inds][1:]  # sort both the thresholds by height
    sorted_heights = np.array(heights, dtype=float)[sort_inds]  # maintain consistency of height with sort indices
    return sum([(sorted_heights[it + 1] - sorted_heights[it])
                * np.heaviside(x - sorted_thresholds[it], 0)
                for it in range(len(sorted_thresholds))]) + \
           sorted_heights[0]

def sym_log(x):
    x_norm = np.sign(x) * np.log10(abs(x) + 1)
    return x_norm

def quiver_norm(x):
    min_arrow_length = 4
    max_arrow_length = 5
    x_new = x
    x_new[np.abs(np.round(x, decimals=10)) == 0] = np.nan
    logs = np.log(np.abs(x_new))
    return np.sign(x) * (np.log10(1 + abs(x) / (10 ** -3)))


def phase_plane(matrix_for_plot, x, y, ss_x=20, ss_y=10, norm=SymLogNorm(linthresh=10 ** -3), ax=None):
    X1, Y1 = np.meshgrid(x, y)  # create 2 matrices
    U = np.zeros_like(matrix_for_plot)
    V = np.array(matrix_for_plot)
    arrow_start_x = 15
    arrow_start_y = 0
    V1 = quiver_norm(V[arrow_start_x::ss_x, arrow_start_y::ss_y])
    ax.quiver(X1[arrow_start_x::ss_x, arrow_start_y::ss_y], Y1[arrow_start_x::ss_x, arrow_start_y::ss_y],
              U[arrow_start_x::ss_x, arrow_start_y::ss_y], V1, width=0.02)


def calcium_step(height, delay_dur=canonical_delay_dur, stim_duration=canonical_stim_dur, end_dur=canonical_end_dur):
    return [0 for it in range(delay_dur)] + [height for it in range(stim_duration)] + [0 for it in range(end_dur)]

class Region:
    def __init__(self, name, bounds, fp, eta, fp_color = None, bar_code_color = None):
        self.name = name
        self.bounds = bounds
        self.fp = fp
        self.eta = eta
        self.fp_color = fp_color
        self.bar_code_color = bar_code_color
        if self.fp_color is None:
            if self.name in default_fp_colors.keys():
                self.fp_color = default_fp_colors[self.name]
            else: self.fp_color = 'k'
        if self.bar_code_color is None:
            if self.name in default_bc_colors.keys():
                self.bar_code_color = default_bc_colors[self.name]
            else: self.bar_code_color = 'k'

    def __str__(self):
        return "Region(name='{}', bounds={}, " \
               "fp={}, eta={}, fp_color='{}', " \
               "bar_code_color='{}')".format(self.name,self.bounds,self.fp,self.eta,
                                             self.fp_color, self.bar_code_color)


def rule_from_regions(eta_dict, fp_dict, name_dict):
    regions = []
    for name, bounds in name_dict.items():
        eta = eta_dict[bounds]
        fp = fp_dict[bounds]
        region = Region(name, bounds, fp, eta)
        regions.append(region)
    return Plasticity_Rule(regions)

class Plasticity_Rule:
    def __init__(self, regions, rule='FPLR'):
        self.rule = rule
        self.regions = sorted(regions, key=lambda region: region.bounds[0])
        self.region_names = [region.name for region in self.regions]
        self.fps = [region.fp for region in self.regions]
        self.thetas = [region.bounds[0] for region in self.regions[1:]]
        self.etas = [region.eta for region in self.regions]
        self.theta_names = [r'$\theta_{' + region.name + '}$' for region in self.regions[1:]]
        self.eta_names = [r'$\eta_{' + region.name + '}$' for region in self.regions[1:]]
        self.fp_names = [r'$\F_{' + region.name + '}$' for region in self.regions[1:]]
        self.region_dict = {region.name: region for region in self.regions}
        self.bar_code_colors = [region.bar_code_color for region in self.regions]
        self.fp_colors = [region.fp_color for region in self.regions]

    def __str__(self):
        # Create a list of strings, each representing a region
        region_strings = []
        for region in self.regions:
            region_strings.append(str(region))
        # Join the strings in the list with a newline character, and return the result
        return 'PlasticityRule(['+'\n'.join(region_strings) + '])'

    def to_pandas(self):
        df = pd.DataFrame()
        for i in range(len(self.regions)-1):
            df.loc[0, self.theta_names[i]] = self.thetas[i]
            df.loc[0, self.fp_names[i]] = self.fps[i+1]
            df.loc[0, self.eta_names[i]] = self.etas[i+1]
        df.columns = ph.latex_to_unicode(df.columns)
        return df


    def fp_from_C(self, C):
        fp_values = [region.fp for region in self.regions]
        thetas = [region.bounds[0] for region in self.regions[1:]]
        return step_function(C, thetas, fp_values)

    def eta_from_C(self, C):
        eta_values = [region.eta for region in self.regions]
        thetas = [region.bounds[0] for region in self.regions[1:]]
        return step_function(C, thetas, eta_values)

    def bar_code_from_C(self, C):
        return step_function(C, self.thetas, range(len(self.thetas) + 1))

    def dw(self, C, w):
        if self.rule == 'FPLR':
            dw = self.FPLR_dw(C, w)
        elif self.rule == 'linear':
            dw = self.linear_dw(C)
        return dw

    def update_weights(self, C, w, dt = 1):
        dw = self.dw(C, w)
        updated_weights = w + dw * dt
        if self.rule == 'linear':
            LB = self.region_dict['N'].fp
            UB = self.region_dict['P'].fp
            updated_weights = np.clip(updated_weights, LB, UB)
        return updated_weights

    def FPLR_dw(self, C, w):
        eta = self.eta_from_C(C)
        fp = self.fp_from_C(C)
        return eta * (fp - w)

    def linear_dw(self, C):
        return self.eta_from_C(C)

    def display(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    def run_plasticity(self, C, w0, dt=1):
        C = np.atleast_2d(C)
        w0 = np.atleast_1d(w0)
        weight_over_time = np.zeros_like(C)
        weight_over_time[:, 0] = w0
        for i in range(1, C.shape[1]):
            Ca_vec = C[:, i - 1]
            delta_w = self.dw(Ca_vec, weight_over_time[:, i - 1])
            weight_over_time[:, i] = weight_over_time[:, i - 1] + (dt * delta_w)
        return np.squeeze(weight_over_time)

    def get_C_range(self, inc=0.005):
        '''Gives a range of calcium values to illustrate rule'''
        min_calc = 0
        max_calc = 1.5 * max(self.thetas)
        C_range = np.arange(min_calc, max_calc, inc)
        return C_range

    def get_w_range(self, inc=0.1):
        '''gives a range of weights between min and max to illutrate rul'''
        eps = 0.1
        if self.rule == 'FPLR':
            min_w = min([region.fp for region in self.regions]) - eps
            max_w = max([region.fp for region in self.regions]) + eps
            w_range = np.arange(min_w, max_w, inc)
        else:
            w_range = np.arange(0 - eps, 1 + eps, inc)
        return w_range

    def fp_and_eta_plot(self, ax=None, title=''):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        C = self.get_C_range()
        FP_line = [self.fp_from_C(c) for c in C]
        eta_line = [self.eta_from_C(c) for c in C]
        ax.plot(C, FP_line, color="k", linewidth=4, label="F.P.")
        ax.set_xlabel(r'$\mathrm{[Ca^{2+}]}$')
        ax.set_ylabel('F(Ca(t))')
        ax.set_yticks(list(self.fps))
        ax.set_xlabel(r'$\mathrm{[Ca^{2+}]}$')
        ax.set_title(title)
        ax.set_xticks(self.thetas)
        ax.set_xticklabels(self.theta_names)
        twinax = ax.twinx()
        twinax.plot(C, eta_line, color='hotpink', label=r'$\eta$' + '(Ca(t))')
        twinax.set_ylabel(r'$\eta$' + '(Ca(t))')
        twinax.set_yticks(list(self.etas))
        twinax.yaxis.label.set_color('hotpink')
        twinax.spines['right'].set_color('hotpink')
        twinax.tick_params(axis='y', colors='hotpink')
        return ax, twinax

    def make_dw_mat(self, w_inc=0.01, C_inc=0.005):
        '''
        gets the weight changes for a grid of different values of calcium and initial weights
        '''
        C_range = self.get_C_range(C_inc)
        w_range = self.get_w_range(w_inc)
        return (np.array([[self.dw(C, w) for C in C_range] for w in w_range])), C_range, w_range

    def dw_line_plot(self, ax=None, show_cbar=True):
        '''
        Calc_Plots instantaneous dw as a function of calcium for different weights
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        dw_mat, C_range, w_range = self.make_dw_mat(w_inc=0.1)
        colors = [mpl.cm.Wistia(x) for x in w_range]
        for n, color in enumerate(colors):
            ax.plot(C_range, dw_mat[n], color=color)
        ax.set_xticks(self.thetas)
        ax.set_xticklabels(self.theta_names)
        ax.set_xlabel(r'$\mathrm{[Ca^{2+}]}$')
        ax.set_ylabel(r'$\mathrm{{\Delta}w}$')
        if show_cbar:
            # norm = mpl.theta_colors.Normalize(vmin=w_range.min(), vmax=w_range.max())
            cmap = mpl.cm.ScalarMappable(cmap=mpl.cm.Wistia)
            cbar = plt.colorbar(cmap, ax=ax)
            cbar.set_label(r'$\mathrm{w}$')

    def canonical_step_stim(self, eps=0.2):
        return [calcium_step(theta + eps) for theta in self.thetas]

    def Ca_stim_plot(self, Ca='cannonical', line_colors=['b', 'r'], theta_colors=['b', 'r'], ax=None):
        '''
        Calc_Plots a matrix of calcium stimulations, indicates fixed points.
        By default plots the standard step stimulus
        '''
        cannonical = 0
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if Ca == 'cannonical':
            cannonical = 1
            Ca = self.canonical_step_stim()
        Ca = np.atleast_2d(Ca)
        for n in range(Ca.shape[0]):
            ax.plot(Ca[n, :], color=line_colors[n])
        for i, theta in enumerate(self.thetas):
            ax.axhline(theta, color=theta_colors[i], linestyle=':', linewidth=1)
        ax.set_yticks(self.thetas, labels=self.theta_names)
        ax.set_ylabel('Time')
        ax.set_ylabel(r'$\mathrm{[Ca^{2+}]}$')
        ax.set_xlabel('Time')
        if cannonical:
            ax.set_xticks([canonical_delay_dur, canonical_delay_dur + canonical_stim_dur], labels=['S', 'E'])

    def weight_change_plot(self, weights, line_colors, fp_colors, line_styles, ax):
        '''
        Given a matrix of weights pver time, plots them, shows fixed points
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        weights = np.atleast_2d(weights)
        for n in range(weights.shape[0]):
            ax.plot(weights[n, :], color=line_colors[n], linestyle=line_styles[n])
        for i, fp in enumerate(self.fps):
            ax.axhline(fp, color=fp_colors[i], linestyle=':', linewidth=1)
        # ax.set_yticks(self.fp_dict.values)
        ax.set_ylabel(r'$\mathrm{w}$')
        ax.set_xlabel('Time')

    def weight_change_plot_from_Ca(self, Ca, w0, line_colors, fp_colors, line_styles, ax=None):
        weights = self.run_plasticity(Ca, w0)
        self.weight_change_plot(weights, line_colors, fp_colors, line_styles, ax)

    def canonical_weight_change_from_Ca(self, ax=None):
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        Ca = self.canonical_step_stim()
        if self.rule == 'FPLR':
            w0 = [self.region_dict["N"].fp, self.region_dict["N"].fp]
        elif self.rule == 'linear':
            w0 = [0.5, 0.5]
        self.weight_change_plot_from_Ca(Ca, w0, ['b', 'r'], ['k', 'b', 'r'], ['-', '-'], ax)
        ax.set_xticks([canonical_delay_dur, canonical_delay_dur + canonical_stim_dur], labels=['S', 'E'])

    def bar_code_imshow(self, C, ax=None, plot_cbar = 1):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        bar_codes = self.bar_code_from_C(np.array(C))
        im = ax.imshow(bar_codes, aspect='auto', cmap=ListedColormap(self.bar_code_colors), origin='lower', vmin=0,
                       vmax=len(self.bar_code_colors))
        ax.set_ylabel(r'$\bf{Plasticity}$' + '\nsyn #')
        if plot_cbar!=0:
            cbar = plt.colorbar(im)
            cbar.set_label('Plasticity')
            cbar.set_ticks(np.arange(len(self.bar_code_colors)) + 0.5, labels=self.region_names)
        return im

    def dw_imshow(self, plot_phase=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        dw_mat, C_range, w_range = self.make_dw_mat(w_inc=0.01, C_inc=0.01)
        bound = max(np.abs(dw_mat.ravel()))
        symlgnrm = SymLogNorm(linthresh=10 ** -3, vmin=-bound, vmax=bound)
        im = ax.imshow(dw_mat, cmap="coolwarm", aspect="auto",
                       extent=[C_range[0], C_range[-1], w_range[0],
                               w_range[-1]],
                       origin="lower", norm=symlgnrm)
        if plot_phase:
            phase_plane(dw_mat, C_range, w_range, ax=ax)
        ax.set_xticks(self.thetas)
        ax.set_xticklabels(self.theta_names)
        ax.set_ylabel(r'$\mathrm{w}$')
        ax.set_xlabel(r'$\mathrm{[Ca^{2+}]}$')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\mathrm{{\Delta}w}$')
        return im


if __name__ == "__main__":
    regions = [Region('N', (-np.inf, 0.5), 0.5, 0),
               Region('D', (0.5, 1), 0, 0.1),
               Region('P', (1, np.inf), 1, 0.1)]
    basic_pl = Plasticity_Rule(regions)
    print(basic_pl.to_pandas())
    # basic_pl.dw_imshow()
    # basic_pl.fp_and_eta_plot()
    # basic_pl.dw_line_plot()
    # basic_pl.Ca_stim_plot()
    # basic_pl.canonical_weight_change_from_Ca()
    # (basic_pl.bar_code_from_C(np.array([np.arange(0,2,0.1), np.arange(1, 3, 0.1)])))
    # basic_pl.bar_code_imshow(np.array([np.arange(0, 2, 0.1)]))
    # plt.show()

    # test step function
    # x = np.arange(0, 2, 0.01)
    # y = step_function(x, thresholds=[0.8, 0.5], heights=[1, 0, 2])
    # plt.plot(x, y)
    # plt.show()
