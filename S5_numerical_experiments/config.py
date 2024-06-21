EXPERIMENT_REFERENCE = "S55weak2" # Reference
RESULTS_FOLDER = "results" # Folder name for saving results
#
import keras, numpy as np, tensorflow as tf # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_2D
EXACT = lambda x,y : (keras.ops.sin(2*x))*(keras.ops.sin(2*y)) # Exact solution function
EXACT_NORM2 = 19.739208802178716 # Norm of the exact solution
SOURCE = lambda x,y : 8 * keras.ops.sin(2 * x) * keras.ops.sin(2 * y) # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 64 # number of trial spanning functions (integer)
M1 = 30 # number of test basis functions
M2 = 30 # number of test basis functions (M1 x M2 must be an integer larger than N)
K1 = 30 # number of integration points (integer larger than M1)
K2 = 30 # number of integration points (integer larger than M2)
KTEST1 = 30 # number of integration points for validation (larger than K1)
KTEST2 = 30 # number of integration points for validation (larger than K2)
IMPLEMENTATION = "weak" # 'weak' or 'ultraweak'
LEARNING_RATE = 0.001
EPOCHS = 100 # Number of iterations
XYPLOT = [tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=30), axis=1)),tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=30), axis=1))] # Domain sample for plotting
LEGEND_LOCATION = "best" # Location for legend in all matplotlib plots