import matplotlib.pyplot as plt

def draw_lines(theta_d, theta_p, alpha_range, gamma_range):
    fig, ax = plt.subplots()

    # Draw horizontal lines
    ax.axhline(y=theta_d, color='b', linestyle='-')
    ax.axhline(y=theta_p, color='r', linestyle='-')

    # Draw vertical lines
    ax.axvline(x=theta_d, color='b', linestyle='-')
    ax.axvline(x=theta_p, color='r', linestyle='-')

    # Draw diagonal lines
    ax.plot([theta_d, 0], [0, theta_d], color='b', linestyle='-')
    ax.plot([theta_p, 0], [0, theta_p], color='r', linestyle='-')

    # Set x and y limits
    ax.set_xlim(alpha_range)
    ax.set_ylim(gamma_range)

    # Set x and y labels
    ax.set_xlabel('alpha')
    ax.set_ylabel('gamma')

    plt.show()

# Usage
draw_lines(0.5, 0.82, [0, 1.5], [0, 1.5])