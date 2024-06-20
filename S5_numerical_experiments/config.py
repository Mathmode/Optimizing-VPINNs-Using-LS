EXPERIMENT_REFERENCE = "S55weak" # Reference
RESULTS_FOLDER = "results" # Folder name for saving results
#
import keras, numpy as np, tensorflow as tf # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_2D
EXACT = lambda x,y : (keras.ops.sin(2*1*x))**2*(keras.ops.sin(2*y))**2*keras.ops.exp((x+y)/2) # Exact solution function
EXACT_NORM2 = 641.3312537604604 # Norm of the exact solution
SOURCE = lambda x,y : -8 * keras.ops.exp((x + y) / 2) * keras.ops.cos(2 * y) ** 2 * keras.ops.sin(2 * 1 * x) ** 2 - 4 * keras.ops.exp((x + y) / 2) * keras.ops.cos(2 * y) * keras.ops.sin(2 * 1 * x) ** 2 * keras.ops.sin(2 * y) - 8 * keras.ops.exp((x + y) / 2) * 1 ** 2 * keras.ops.cos(2 * 1 * x) ** 2 * keras.ops.sin(2 * y) ** 2 - 4 * keras.ops.exp((x + y) / 2) * 1 * keras.ops.cos(2 * 1 * x) * keras.ops.sin(2 * 1 * x) * keras.ops.sin(2 * y) ** 2 + 15 / 2 * keras.ops.exp((x + y) / 2) * keras.ops.sin(2 * 1 * x) ** 2 * keras.ops.sin(2 * y) ** 2 + 8 * keras.ops.exp((x + y) / 2) * 1 ** 2 * keras.ops.sin(2 * 1 * x) ** 2 * keras.ops.sin(2 * y) ** 2 # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 32 # number of trial spanning functions (integer)
M1 = 16 # number of test basis functions
M2 = 16 # number of test basis functions (M1 x M2 must be an integer larger than N)
K1 = 64 # number of integration points (integer larger than M1)
K2 = 64 # number of integration points (integer larger than M2)
KTEST1 = 256 # number of integration points for validation (larger than K1)
KTEST2 = 256 # number of integration points for validation (larger than K2)
IMPLEMENTATION = "weak" # 'weak' or 'ultraweak'
LEARNING_RATE = 0.001
EPOCHS = 200 # Number of iterations
XYPLOT = [tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=256), axis=1)),tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=256), axis=1))] # Domain sample for plotting
LEGEND_LOCATION = "best" # Location for legend in all matplotlib plots