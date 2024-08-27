# import glob
# import os
#
# # Use glob to get a list of files that match the pattern "Fig*.py"
# # (i.e. files starting with "Fig" followed by zero or more characters and ending with ".py")
# from matplotlib import pyplot as plt
#
# import param_helpers
#
# file_list = glob.glob("Fig*.py")
#
# # Iterate over the list of files
# for file in file_list:
#     # Use os.system() to run each file.
#     # The `python` command is used to run a Python script from the command line.
#     # The `-B` flag prevents Python from writing .pyc files.
#     # The `-u` flag forces Python to flush its output buffers after each write operation.
#     print(str(file))
#     os.system(f"python -B -u {file}")
#     plt.close('all')
#
# param_helpers.param_concat()
import glob
import os
import subprocess
from matplotlib import pyplot as plt
import param_helpers
import sys

# Use the current Python interpreter
python_executable = sys.executable

# Use glob to get a list of files that match the pattern "Fig*.py"
file_list = glob.glob("Fig*.py")

# Iterate over the list of files
for file in file_list:
    print(str(file))
    subprocess.run([python_executable, '-B', '-u', file])
    plt.close('all')

param_helpers.param_concat()
