

def create_config(global_variables):
    
   EXPERIMENT_REFERENCE, RESULTS_FOLDER, PACKAGES, EXACT, SOURCE, A, B, N, M, K, KTEST, IMPLEMENTATION, SAMPLING, LEARNING_RATE, EPOCHS, XPLOT = global_variables

   
   config_content = f"""EXPERIMENT_REFERENCE = "{EXPERIMENT_REFERENCE}" # Reference
RESULTS_FOLDER = "{RESULTS_FOLDER}" # Folder name for saving results
#
{PACKAGES} # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_1D
EXACT = {EXACT} # Exact solution function
SOURCE = {SOURCE} # 'Printed' in the script as a function
A = {float(A)} # Left end-point of the domain of integration
B = {float(B)} # Right end-point of the domain of integration
N = {int(N)} # number of trial spanning functions (integer)
M = {int(M)} # number of test basis functions (integer larger than N)
K = {int(K)} # number of integration points (integer larger than M)
KTEST = {int(KTEST)} # number of integration points for validation (larger than K)
IMPLEMENTATION = "{IMPLEMENTATION}" # 'weak' or 'ultraweak'
SAMPLING = "{SAMPLING}" # 'uniform' or 'exponential'
LEARNING_RATE = {float(LEARNING_RATE)}
EPOCHS = {EPOCHS} # Number of iterations
XPLOT = {XPLOT} # Domain sample for plotting"""

   # Write the new content to config.py
   with open('config.py', 'w') as file:
           file.write(config_content)