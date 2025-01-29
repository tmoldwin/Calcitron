import numpy as np
from matplotlib import pyplot as plt
from calcitron import Calcitron
import Calcitron_functions as cf

P = 50
n_synapses = 10

W_0 = np.linspace(0, 5.5, n_synapses)
Z = np.linspace(0,cf.theta_p + 0.2, P)
X = np.ones((P, n_synapses))
calc1 = Calcitron(0.2,0,0,1,n_synapses)
calc1.weights = np.array(W_0)


for prot in [0]:
    calc1.train(X, Z, protocol = prot)
    plt.imshow(calc1.delta_w.T, cmap="coolwarm", aspect="auto", extent=[0, Z[-1], 0, W_0[-1]],
                        origin="lower")
l = plt.ylabel(r'$\Delta$' + 'w', rotation = 0, labelpad= 10)
plt.xlabel('Ca global')
plt.title('Increasing supervising signal')
plt.show()