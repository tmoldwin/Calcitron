import numpy as np
import pandas as pd

def make_label_map(X,y):
    X = np.atleast_2d(X)
    label_map = {tuple(X[i]): y[i] for i in range(X.shape[0])}
    return label_map

class Supervisor:
    def __init__(self):
        self.Z_p = 0
        self.Z_d = 0
        self.Z = 0
        self.eta_b = 0
        pass

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        pass

    def delta_bias(self, pattern, pattern_index, y_hat_binary, y_hat):
        return 0

    def to_pandas(self):
        # Create a dictionary of the values
        data = {
            'Z_P': round(self.Z_p, 2),
            'Z_D': round(self.Z_d, 2),
            'Z': round(self.Z, 2),
            'η_b': round(self.eta_b, 2)
        }

        # Remove zero values
        data = {k: v for k, v in data.items() if v != 0}

        # Convert to DataFrame
        df = pd.DataFrame(data, index=[0])

        return df


class null_supervisor(Supervisor):
    def __init__(self):
        super().__init__()
    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        if y_hat_binary == 1:
            label = 'TP'
        else: label = 'TN'
        return 0, label #returns Z, +1 for fp, -1 for fn, 0 for correct

class signal_noise_supervisor(Supervisor):
    def __init__(self, signals):
        super().__init__()
        self.signals = np.atleast_2d(signals)
        self.Z = 0

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        target = np.any(np.all(pattern == self.signals, axis=1))
        if y_hat_binary == 1:
            if target:
                label = 'TP'
            else:
                label = 'FP'
        elif y_hat_binary == 0:
            if target:
                label = 'FN'
            else:
                label = 'TN'
        return self.Z, label

class one_shot_flip_flop_supervisor(Supervisor):
    def __init__(self, y):
        super().__init__()
        self.y = y
        self.Z = 1

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        if y_hat_binary == 1:
            label = 'TP'
        else:
            label = 'TN'
        return self.y[pattern_index], label

class target_perceptron_supervisor(Supervisor):
    def __init__(self, Zp = 0.3, X = None, y = None, eta_b = 0.1):
        super().__init__()
        self.eta_b = eta_b
        self.Z_p = Zp
        if not X is None:
            self.label_map = make_label_map(X, y)

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        target = self.label_map[tuple(pattern)]
        if y_hat_binary == 1:
            if target:
                label = 'TP'
            else:
                label = 'FP'
        elif y_hat_binary == 0:
                if target:
                    label = 'FN'
                else:
                    label = 'TN'
        return target*self.Z_p, label

    def delta_bias(self, pattern_index, pattern, y_hat_binary, y_hat):
        return self.eta_b*(self.label_map[tuple(pattern)]-y_hat_binary)

class critic_perceptron_supervisor(Supervisor):
    def __init__(self, Z_d = 0.2, Z_p = 0.5, X = None, y = None, eta_b=0.1):
        super().__init__()
        self.Z_d = Z_d
        self.Z_p = Z_p
        self.eta_b = eta_b
        if not X is None:
            self.label_map = make_label_map(X, y)

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        target = self.label_map[tuple(pattern)]
        Z = 0
        if y_hat_binary == 1:
            if target:
                label = 'TP'
            else:
                label = 'FP'
                Z = self.Z_d
        elif y_hat_binary == 0:
            if target:
                label = 'FN'
                Z = self.Z_p
            else:
                label = 'TN'
        return Z, label

    def delta_bias(self, pattern_index, pattern, y_hat_binary, y_hat):
        return self.eta_b * (self.label_map[tuple(pattern)] - y_hat_binary)

class homeostatic_supervisorPD(Supervisor):
    def __init__(self, min_target, max_target, Z_d, Z_p):
        super().__init__()
        self.min_target = min_target
        self.max_target = max_target
        self.Z_d = Z_d
        self.Z_p = Z_p

    def supervise(self, pattern_index, pattern, y_hat_binary, y_hat):
        if y_hat > self.max_target:
            return self.Z_d, 'H'
        elif y_hat < self.min_target:
            return self.Z_p, 'L'
        else:
            return 0, 'M'



