#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

This script is currently optimized to be executed in 'David' or 'Goliat' servers.

@author: curiarteb
"""

import subprocess, os, csv, shutil
import numpy as np

##### Configuration variables #####

# Path to the Python binary that will run 'inner.py'
RUN_PATH = './checkrun.sh'
# String to identify the line reporting FLOPs. CPU-dependent!!
FLOPS_ID = 'fp_ret_sse_avx_ops.all'
# Perf command for counting flops.
# 1000 is the number of miliseconds per FLOP calculation.
PERF = ['perf','stat','-I','1000','-e', FLOPS_ID]
# Path to the inner script
INNER = 'innerAD.py'
# Perf command. CPU-dependent!!
COMMAND = PERF + [RUN_PATH, INNER]
# Results folder
RESULTS_FOLDER = 'results'
# Error record txt
ERROR_FILE = RESULTS_FOLDER + "/error_record.txt"

# To give permission to RUN_PATH for execution
subprocess.run(['chmod','a+x', f'{RUN_PATH}'])

# Create 'results' folder    
if os.path.exists(RESULTS_FOLDER):
    shutil.rmtree(RESULTS_FOLDER)
    os.makedirs(RESULTS_FOLDER)
else:
    os.makedirs(RESULTS_FOLDER)

def subprocess2measureflops(input_dim, output_dim, checkpoint):
    
    process_command = COMMAND + [str(input_dim),str(output_dim),checkpoint]

    sp = subprocess.run(process_command, capture_output=True, text=True, check=True)

    # In case the subprocess terminates with error (see 'checkrun.sh'),
    # we print an error message and save the 'stdout' and 'stderr'. 
    if 'RUNERROR' in sp.stdout:
        string = " ".join(process_command)
        print(f"Subprocess '{string}' failed!")
        print(f"See '{ERROR_FILE}' for the 'stdout' and the 'stderr' of the failed execution.")
        with open(ERROR_FILE, 'a') as f:
            f.write(f"\nSUBPROCESS: '{string}'\n")
            f.write("\n\nSTDOUT:\n")
            f.write(sp.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(sp.stderr)
            f.write("\n*****************************************************************")
            f.write("*****************************************************************")
            f.write("*****************************************************************")
            f.write("*****************************************************************")
                
        return np.nan
    
    # If the execution is correct, we take the FLOPs of the execution
    else:        
        #Not sure why output is sent to stderr
        out = sp.stderr.split('\n')
        #Look for the interesting lines
        lines = [l for l in out if FLOPS_ID in l]
        # Add all the contributions in the list of flops
        flops_list = [int(line.split()[1].replace('.', '')) for line in lines]
        flops = sum(flops_list)
        return flops
    
# Create three csv for different data (open, intialize, and close)
flops = RESULTS_FOLDER + '/flops_AD.csv'
ratios = RESULTS_FOLDER + '/ratios_AD.csv'

headers1 = ['input_dim','output_dim','eval','fwd','bwd']
headers2 = ['input_dim','output_dim','fwd','bwd']

with open(flops, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers1)
    writer.writeheader()

with open(ratios, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers2)
    writer.writeheader()
    
# Input and output dimensions for experimentation
sizes = [2**k for k in range(10)]

# We perform the experimentation and compute the checkpoint-by-checkpoint 
# subtractions to extract the interested FLOP measurements.
for i in sizes:
    for o in sizes:
        print("\nINPUT:", i, "- OUTPUT:", o)
        
        post = subprocess2measureflops(i, o, "baseline")
        baseline = post
        prev = post
        print("baseline")
        
        post = subprocess2measureflops(i, o, "eval")
        evaluation = post - prev
        prev = post
        print("eval")
        
        post = subprocess2measureflops(i, o, "fwd")
        fwd = post - prev
        ratio_fwd = fwd/evaluation
        prev = post
        print("fwd")
        
        post = subprocess2measureflops(i, o, "bwd")
        bwd = post - prev
        ratio_bwd = bwd/evaluation
        prev = post
        print("bwd")
        
        flops_dict = {'input_dim':i,
                      'output_dim':o,
                      'eval':"{:.2e}".format(evaluation),
                      'fwd':"{:.2e}".format(fwd),
                      'bwd':"{:.2e}".format(bwd)
                      }
        
        ratios_dict = {'input_dim':i,
                      'output_dim':o,
                      'fwd':"{:.2e}".format(ratio_fwd),
                      'bwd':"{:.2e}".format(ratio_bwd)
                      }
        
        # Load data in the csv-s (open, add, and close)
        with open(flops, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers1)
            writer.writerow(flops_dict)
        
        with open(ratios, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers2)
            writer.writerow(ratios_dict)
        
        print(f"line in {flops} and {ratios}")
        

print("\n'outerFLOPs.py' run finished!")

# We remove the ERROR_FILE if it is empty. Otherwise, we leave it.
if os.path.exists(ERROR_FILE):
    if os.path.getsize(ERROR_FILE) == 0:
        os.remove(ERROR_FILE)