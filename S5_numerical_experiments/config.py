EXPERIMENT_REFERENCE = "S54ultraweak_M128" # Reference
RESULTS_FOLDER = "results" # Folder name for saving results
#
import numpy as np, tensorflow as tf # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_1D
EXACT = lambda x : x**0.7*(3.141592653589793-x) # Exact solution function
EXACT_NORM2 = 11.375919271175881 # Norm of the exact solution
SOURCE = lambda x : 0.7*x**(0.7-2)*(0.7*x-3.141592653589793*0.7+x+3.141592653589793) # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 16 # number of trial spanning functions (integer)
M = 128 # number of test basis functions (integer larger than N)
K = 4096 # number of integration points (integer larger than M)
KTEST = 32768 # number of integration points for validation (larger than K)
IMPLEMENTATION = "ultraweak" # 'weak' or 'ultraweak'
SAMPLING = "exponential" # 'uniform' or 'exponential'
LEARNING_RATE = 0.001
EPOCHS = 1000 # Number of iterations
XPLOT = tf.convert_to_tensor(np.expand_dims(np.linspace(A+10**(-4), B, num=1000), axis=1)) # Domain sample for plotting
LEGEND_LOCATION = "best" # Location for legend in all matplotlib plots