import numpy as np
from matplotlib import pyplot as plt
from calcitron import Calcitron
import Calcitron_functions as cf
import creating_colormap as cc
import seaborn as sns
from matplotlib import cm
import matplotlib.colors as mcolors
cmap=cm.get_cmap('coolwarm')
P = 25
n_synapses = 100

X = np.ones((P, n_synapses))
Z = np.zeros(P)
Z[2] = 0.6
Z_0 = np.zeros(P)
W_0 = np.linspace(0, 5.5, n_synapses)
calc0 = Calcitron(0.2,0,0,1,n_synapses)
calc1 = Calcitron(0.2,0,0,1,n_synapses)


calc1.weights = np.array(W_0)
calc0.weights = np.array(W_0)
fig_overtime, axes_graphs = plt.subplots(3,2, figsize = (7,8))

cmap0 = mcolors.LinearSegmentedColormap.from_list('mycmap', [(0, 'white'), (0.09, 'yellow'),
                                                    (0.17, 'orange'),
                                                    (1, '#760404')])
cmap1 = mcolors.LinearSegmentedColormap.from_list('mycmap', [(0, 'white'), (0.04, 'yellow'),(0.12, 'orange'),
                                                             (0.3,'red'),(1, '#760404')])
# cmap1 = mcolors.LinearSegmentedColormap.from_list('mycmap', [(0, 'white'), (0.08, 'yellow'),
#                                                     (0.1, 'orange'), (0.25, 'red'),
#                                                     (0.8, '#760404'), (1, 'black')])
delta_cmap0 = mcolors.LinearSegmentedColormap.from_list('mycmap', [(0, 'blue'),
                                                    (0.1, 'white'),
                                                    (1, 'red')])
for prot in [0]:
    calc0.train(X, Z_0, protocol=prot, eta=0.5, hard_threshold=0, k=0.25)
    # colormap0 = cm.LinearColormap(theta_colors=['blue', 'white', 'red'], index=[np.amin(calc0.delta_w), 0, np.amax(calc0.delta_w)]
    #                               , vmin=np.amin(calc0.delta_w),
    #                              vmax=np.amax(calc0.delta_w))
    offset0 = mcolors.TwoSlopeNorm(vmin=np.amin(calc0.delta_w),vcenter=0., vmax=np.amax(calc0.delta_w))
    #plot_data, asym_cmap0 = cc.asymmetric_cmap(calc0.delta_w, cmap, ref_point=0)[:2]
    axes_graphs[0,0].imshow(calc0.delta_w.T, cmap=delta_cmap0, aspect="auto", extent=[0, P, 0, W_0[-1]],
                          origin="lower")
    axes_graphs[2,0].imshow(calc0.weight_history.T, cmap=cmap0, aspect="auto", extent=[0, P, 0, W_0[-1]],
                            origin="lower")

    calc1.train(X, Z, protocol = prot, eta = 0.5, hard_threshold=0, k=0.25)
    # colormap1 = cm.LinearColormap(theta_colors=['blue', 'white', 'red'], index=[np.amin(calc1.delta_w), 0, np.amax(calc1.delta_w)],
    #                               vmin=np.amin(calc1.delta_w), vmax=np.amax(calc1.delta_w))
    offset1 = mcolors.TwoSlopeNorm(vmin=np.amin(calc1.delta_w), vcenter=0., vmax=np.amax(calc1.delta_w))
    #plot_data, asym_cmap1 = cc.asymmetric_cmap(calc1.delta_w, cmap, ref_point=0)[:2]
    axes_graphs[0,1].imshow(calc1.delta_w.T, cmap= delta_cmap0, aspect="auto", extent=[0, P, 0,W_0[-1]],
                        origin="lower" )
    axes_graphs[2,1].imshow(calc1.weight_history.T, cmap=cmap1, aspect="auto", extent=[0, P, 0,W_0[-1]],
                            origin="lower" )

    #indices = [10,25,40, 60,70, 72, 80, 90]
    indices = range(3,n_synapses, 10)
    # lines = [calc0.weight_history.T[i,:] for i in indices]
    times = list(range(P))
    # (lines)
    # (times)
    # axes_graphs[1, 0].plot(times,lines)
    for index in indices:
        axes_graphs[1, 0].plot(times,calc0.weight_history.T[index])
        axes_graphs[1, 1].plot(times,calc1.weight_history.T[index])




axes_graphs[0,0].set_title('Only Local Ca')
axes_graphs[0,0].set_ylabel(r'$w_0$', rotation = 0, labelpad=10)
axes_graphs[0,0].set_xlabel('Time')
axes_graphs[2,0].set_ylabel(r'$w_0$', rotation = 0, labelpad=10)
axes_graphs[2,0].set_xlabel('Time')
axes_graphs[0,1].set_title('With Global Supervising Signal at t=2')
axes_graphs[0,1].set_ylabel(r'$w_0$', rotation = 0, labelpad=10)
axes_graphs[0,1].set_xlabel('Time')
axes_graphs[2,1].set_ylabel(r'$w_0$', rotation = 0, labelpad=10)
axes_graphs[2,1].set_xlabel('Time')
axes_graphs[1, 0].set_ylim([0,10])
axes_graphs[1, 0].set_xlim([0,P-1])
axes_graphs[1, 0].set_xlabel('Time')
axes_graphs[1, 0].set_ylabel('w', rotation=0)
axes_graphs[1, 1].set_ylim([0,10])
axes_graphs[1, 1].set_xlim([0,P-1])
axes_graphs[1, 1].set_xlabel('Time')
axes_graphs[1, 1].set_ylabel('w', rotation=0)
#plt.tight_layout()
fig_overtime.tight_layout()
#plt.savefig(r'C:\Users\kelly\Dropbox\the Calcitron\presentation figures\constant_calcium')
plt.show()
