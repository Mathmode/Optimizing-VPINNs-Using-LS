

def create_config(global_variables):
    
   EXPERIMENT_REFERENCE, RESULTS_FOLDER, PACKAGES, EXACT, EXACT_NORM2, SOURCE, A, B, N, M1, M2, K1, K2, KTEST1, KTEST2, IMPLEMENTATION, LEARNING_RATE, EPOCHS, XYPLOT, LEGEND_LOCATION = global_variables

   
   config_content = f"""EXPERIMENT_REFERENCE = "{EXPERIMENT_REFERENCE}" # Reference
RESULTS_FOLDER = "{RESULTS_FOLDER}" # Folder name for saving results
#
{PACKAGES} # Necessary packages (if needed)
#
# Gobal parameters for the scripts in SCR_2D
EXACT = {EXACT} # Exact solution function
EXACT_NORM2 = {float(EXACT_NORM2)} # Norm of the exact solution
SOURCE = {SOURCE} # 'Printed' in the script as a function
A = {float(A)} # Left end-point of the domain of integration
B = {float(B)} # Right end-point of the domain of integration
N = {int(N)} # number of trial spanning functions (integer)
M1 = {int(M1)} # number of test basis functions
M2 = {int(M2)} # number of test basis functions (M1 x M2 must be an integer larger than N)
K1 = {int(K1)} # number of integration points (integer larger than M1)
K2 = {int(K2)} # number of integration points (integer larger than M2)
KTEST1 = {int(KTEST1)} # number of integration points for validation (larger than K1)
KTEST2 = {int(KTEST2)} # number of integration points for validation (larger than K2)
IMPLEMENTATION = "{IMPLEMENTATION}" # 'weak' or 'ultraweak'
LEARNING_RATE = {float(LEARNING_RATE)}
EPOCHS = {EPOCHS} # Number of iterations
XYPLOT = {XYPLOT} # Domain sample for plotting
LEGEND_LOCATION = "{LEGEND_LOCATION}" # Location for legend in all matplotlib plots"""

   # Write the new content to config.py
   with open('config.py', 'w') as file:
           file.write(config_content)