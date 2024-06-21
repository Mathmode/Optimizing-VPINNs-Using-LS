EXPERIMENT_REFERENCE = "S52weak" # Reference
RESULTS_FOLDER = "results" # Folder name for saving results
#
import keras, numpy as np, tensorflow as tf # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_1D
EXACT = lambda x : keras.ops.sin(4*x)*keras.ops.sin(x/2) # Exact solution function
EXACT_NORM2 = 12.762720155208534 # Norm of the exact solution
SOURCE = lambda x : -4*keras.ops.cos(4*x)*keras.ops.cos(x/2)+65/4*keras.ops.sin(4*x)*keras.ops.sin(x/2) # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 16 # number of trial spanning functions (integer)
M = 32 # number of test basis functions (integer larger than N)
K = 1024 # number of integration points (integer larger than M)
KTEST = 8192 # number of integration points for validation (larger than K)
IMPLEMENTATION = "weak" # 'weak' or 'ultraweak'
SAMPLING = "uniform" # 'uniform' or 'exponential'
LEARNING_RATE = 0.001
EPOCHS = 1000 # Number of iterations
XPLOT = tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=8192), axis=1)) # Domain sample for plotting
LEGEND_LOCATION = "best" # Location for legend in all matplotlib plots