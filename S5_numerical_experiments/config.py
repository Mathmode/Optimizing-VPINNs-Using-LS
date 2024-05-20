# Necessary packages (if needed)
import keras
#
# Gobal parameters for the scripts in SCR_1D
EXACT = lambda x : keras.ops.sin(4*x)*keras.ops.sin(x/2) # 'Printed' in the script as a function
SOURCE = lambda x : -4*keras.ops.cos(4*x)*keras.ops.cos(x/2)+65/4*keras.ops.sin(4*x)*keras.ops.sin(x/2) # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 16 # number of trial spanning functions (integer)
M = 32 # number of test basis functions (integer larger than N)
K = 512 # number of integration points (integer larger than M)
KTEST = 4096 # number of integration points for validation (larger than K)
IMPLEMENTATION = "weak" # 'weak' or 'ultraweak'
SAMPLING = "uniform" # 'uniform' or 'exponential'
LEARNING_RATE = 0.01