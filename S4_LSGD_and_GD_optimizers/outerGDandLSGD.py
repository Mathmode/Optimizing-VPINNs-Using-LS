#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Last edited on May, 2024

@author: curiarteb
'''

##### Configuration variables #####

# Path to the Python binary that will run the 'outer{...}.py' scripts
RUN_PATH = './checkrun.sh'
PYTHON_PATH = 'python'
RESULTS_FOLDER = 'results'
ERROR_FILE = RESULTS_FOLDER + '/error_record.txt'

import os, shutil, subprocess
import pandas as pd

# To give permission to RUN_PATH to run the 'inner{...}.py' scripts
subprocess.run(['chmod','a+x', RUN_PATH])
    
# Create an empty RESULTS_FOLDER (or empty it if it exists)
if os.path.exists(RESULTS_FOLDER):
    shutil.rmtree(RESULTS_FOLDER)
    os.makedirs(RESULTS_FOLDER)
else:
    os.makedirs(RESULTS_FOLDER)

# Create an empty ERROR_FILE
with open(ERROR_FILE, 'w') as file:
    pass
    
def run_outer_script(outer_script):

    sp = subprocess.Popen([PYTHON_PATH,'-u', outer_script, RESULTS_FOLDER, ERROR_FILE],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in sp.stdout:
        print(line, end='')

    for line in sp.stderr:
        print(line, end='')

    # Espera a que el proceso termine
    sp.wait()

##### We execute the set of runs for the GD optimizer #####
run_outer_script('outerGD.py')

##### We execute the set of runs for the LS/GD optimizer #####
run_outer_script('outerLSGD.py')

# We remove the ERROR_FILE if it is empty. Otherwise, we leave it.
if os.path.getsize(ERROR_FILE) == 0:
    os.remove(ERROR_FILE)

##### We compute the final ratios between LS/GD and GD #####
# Load the data of the previous results 
gd_data = pd.read_csv(RESULTS_FOLDER + "/flops_GD.csv")
lsgd_data = pd.read_csv(RESULTS_FOLDER + "/flops_LSGD.csv")

# Merge the above restricting to 'N' and 'imple' keys
merged_data = pd.merge(gd_data, lsgd_data,
                       on=["N", "imple"],
                       suffixes=("_GD", "_LSGD"))

# Calculate the ratios for each 'N'
merged_data["ratio"] = (merged_data["total_LSGD"] / merged_data["total_GD"])

# Redefine the row order
order = ["ultraweak", "weakfwd", "weakbwd"]

# Pivot the data so that 'N' indexes columns and 'imple' indexes rows, respecting the order
pivoted_data = merged_data.pivot(index="imple",
                                 columns="N",
                                 values="ratio").loc[order]

# Save results in a new CSV
pivoted_data.to_csv(RESULTS_FOLDER + "/ratios_LSGDvsGD.csv")


