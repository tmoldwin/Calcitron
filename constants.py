import os
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
PANEL_LABEL_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

PLOT_FOLDER = 'Calc_Plots/'
calcitron_directory = 'C:/Code/Calcitron1'
# PAPER_PLOT_FOLDER = os.path.join(dropbox_directory, 'Calcitron_Paper/', 'Figures/', 'FinalTIFFs/')

# Ensure the directory exists
# if not os.path.exists(PAPER_PLOT_FOLDER):
#     os.makedirs(PAPER_PLOT_FOLDER)
DATA_FOLDER = 'Data/'
PARAMS_FOLDER = 'Params/'