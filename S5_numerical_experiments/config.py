# Necessary packages (if needed)

#
# Gobal parameters for the scripts in SCR_1D
EXACT = lambda x : x**0.6*(3.141592653589793-x) # Exact solution function
DEXACT = lambda x : -x**(0.6-1)*(0.6*(x-3.141592653589793)+x) # Derivative of the exact solution function
SOURCE = lambda x : 0.6*x**(0.6-2)*(0.6*x-3.141592653589793*0.6+x+3.141592653589793) # 'Printed' in the script as a function
A = 0.0 # Left end-point of the domain of integration
B = 3.141592653589793 # Right end-point of the domain of integration
N = 64 # number of trial spanning functions (integer)
M = 128 # number of test basis functions (integer larger than N)
K = 4096 # number of integration points (integer larger than M)
KTEST = 32768 # number of integration points for validation (larger than K)
IMPLEMENTATION = "ultraweak" # 'weak' or 'ultraweak'
SAMPLING = "exponential" # 'uniform' or 'exponential'
LEARNING_RATE = 0.01