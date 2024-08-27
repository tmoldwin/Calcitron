##This is an extremely overkill way to find where
# to place the annotations in the line plots from Figure 3.


import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_distance_grid(X, Y, lines):
    Z = np.full(X.shape, np.inf)
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            for k in range(lines.shape[0]):
                a, b, c = lines[k]
                distance = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
                if distance < Z[i, j]:
                    Z[i, j] = distance
    return Z

def find_peaks(Z_smoothed, grid_size):
    peaks = maximum_filter(Z_smoothed, size=(grid_size // 10, grid_size // 10)) == Z_smoothed
    peak_indices = np.where(peaks)
    return peak_indices

def example_find_peaks(lines, grid_size=200, sigma=4, plot = False):
    # Define bounds of the plane
    x_min, x_max = 0, 1.2
    y_min, y_max = 0, 1.2

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Convert lines to NumPy array
    lines_array = np.array(lines)

    # Compute distance grid
    Z = compute_distance_grid(X, Y, lines_array)

    # Apply Gaussian smoothing
    Z_smoothed = gaussian_filter(Z, sigma=sigma)

    # Find peaks
    peak_indices = find_peaks(Z_smoothed, grid_size)
    if plot:
        plot_peaks(X, Y, Z_smoothed, peak_indices, lines, grid_size=grid_size)
        plt.show()

    return X[peak_indices], Y[peak_indices]

def plot_peaks(X, Y, Z_smoothed, peak_indices, lines, grid_size=200):
    # Define bounds of the plane
    x_min, x_max = 0, 1.2
    y_min, y_max = 0, 1.2

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    # Create a figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 2D Heatmap
    im = ax1.imshow(Z_smoothed, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='jet', alpha=0.8)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title('2D Smoothed Heatmap: Distance to Closest Line')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im, ax=ax1, label='Distance to Closest Line')

    # Plot the lines on the heatmap
    for line in lines:
        a, b, c = line
        if b != 0:
            y_vals = -(a * x_grid + c) / b
            ax1.plot(x_grid, y_vals, 'white')
        else:
            x_val = -c / a
            ax1.axvline(x=x_val, color='white')

    # Plot peaks
    ax1.plot(X[peak_indices], Y[peak_indices], 'ko', markersize=8)  # Black dots for peaks

    # 3D Surface Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_smoothed, cmap='jet', edgecolor='none')
    ax2.set_title('3D Smoothed Surface Plot: Distance to Closest Line')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Distance to Closest Line')

    # Plot the lines on the surface plot
    for line in lines:
        a, b, c = line
        if b != 0:
            y_vals = -(a * x_grid + c) / b
            z_vals = np.zeros_like(y_vals)
            ax2.plot(x_grid, y_vals, z_vals, color='white')
        else:
            x_val = -c / a
            z_vals = np.zeros_like(y_grid)
            ax2.plot([x_val] * len(y_grid), y_grid, z_vals, color='white')

    plt.tight_layout()



if __name__ == '__main__':
    # Example usage
    theta_ds = [0.3, 0.7]
    theta_p = 1.0
    mx = 1.2

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, theta_d in zip(axes, theta_ds):
        lines = [
            [1, 0, -theta_d],  # x = theta_d
            [1, 0, -theta_p],  # x = theta_p
            [0, 1, -theta_d],  # y = theta_d
            [0, 1, -theta_p],  # y = theta_p
            [1, 1, -theta_d],  # x + y = theta_d
            [1, 1, -theta_p],  # x + y = theta_p
            [1, 0, 0],  # x = 0 (left boundary)
            [0, 1, 0],  # y = 0 (bottom boundary)
            [1, 0, -mx],  # x = mx (right boundary)
            [0, 1, -mx],  # y = mx (top boundary)
        ]

        # Define line colors
        line_colors = {
            (1, 0, -theta_d): 'blue',
            (1, 0, -theta_p): 'red',
            (0, 1, -theta_d): 'blue',
            (0, 1, -theta_p): 'red',
            (1, 1, -theta_d): 'blue',
            (1, 1, -theta_p): 'red',
            (1, 0, 0): 'black',
            (0, 1, 0): 'black',
            (1, 0, -mx): 'black',
            (0, 1, -mx): 'black',
        }

        # Find peaks
        peak_x, peak_y = example_find_peaks(lines)

        # Create peak annotations
        grid_size = 100
        x_min, x_max = 0, 1.2
        y_min, y_max = 0, 1.2
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = compute_distance_grid(X, Y, np.array(lines))
        Z_smoothed = gaussian_filter(Z, sigma=4.8)
        peak_indices = find_peaks(Z_smoothed, grid_size)
        plot_peaks(X, Y, Z_smoothed, peak_indices, lines, grid_size=grid_size)
        plt.show()
        # annotations = create_peak_annotations(X, Y, peak_indices, theta_d, theta_p)
        #
        # # Plot lines with annotations
        # plot_lines_with_annotations(ax, lines, annotations, theta_d, theta_p, line_colors)
        #
        # axes[1].set_yticks([]); axes[1].set_ylabel(''); axes[1].tick_params(labelleft=False)
        # plt.tight_layout()
        # plt.show()
