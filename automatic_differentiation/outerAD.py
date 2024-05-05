#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2024

Script to measure FLOPs using 'perf' as underlying tool.

This script is currently optimized to be executed in the David server.

@author: curiarteb
"""

import subprocess, os, csv, shutil
import numpy as np

##### Configuration variables #####
# Path to the instruction that will run 'innerAD.py'
RUN_PATH = './checkrun.sh'
# Path to the inner script
INNER = 'innerAD.py'
# String to identify the line reporting FLOPS. CPU-dependent!
FLOPS_ID = 'fp_ret_sse_avx_ops.all'
# Perf command
COMMAND = f'perf stat -I 1000 -e {FLOPS_ID} {RUN_PATH} {INNER}'
# Results folder
FOLDER_NAME = 'results'

# To give permission to RUN_PATH for execution
subprocess.run(['chmod','a+x', f'{RUN_PATH}'])

# Create 'results' folder    
if os.path.exists(FOLDER_NAME):
    shutil.rmtree(FOLDER_NAME)
    os.makedirs(FOLDER_NAME)
else:
    os.makedirs(FOLDER_NAME)

def subprocess2measureflops(input_dim, output_dim, checkpoint, error_file= FOLDER_NAME + "/stdout_stderr.txt"):

    sp = subprocess.run([f'{COMMAND} {input_dim} {output_dim} {checkpoint}'], 
                        shell=True, capture_output=True,
                        text=True, check=True)

    # In case the subprocess terminates with error (see 'checkrun.sh'),
    # we print an error message and save the 'stdout' and 'stderr'. 
    if 'RUNERROR' in sp.stdout:
        print(f"Subprocess '{COMMAND} {input_dim} {output_dim} {checkpoint}' failed!")
        print(f"See '{error_file}' for the 'stdout' and the 'stderr' of the failed execution.")
        with open(error_file, 'a') as f:
            f.write(f"SUBPROCESS: '{COMMAND} {input_dim} {output_dim} {checkpoint}'\n")
            f.write("STDOUT:\n")
            f.write(sp.stdout)
            f.write("\nSTDERR:\n")
            f.write(sp.stderr)
            f.write("\n*****************************************************************\n")
        f.close()
                
        return np.nan
    
    # If the execution is correct, we take the FLOPs of the execution
    else:        
        # If no error, remove the output file if it exists from a previous execution
        if os.path.exists(error_file):
            os.remove(error_file)
        
        #Not sure why output is sent to stderr
        out = sp.stderr.split('\n')
        #Look for the interesting lines
        lines = [l for l in out if FLOPS_ID in l]
        # Add all the contributions in the list of flops
        flops_list = [int(line.split()[1].replace('.', '')) for line in lines]
        flops = sum(flops_list)
        return flops
    
# Create three csv for different data (open, intialize, and close)
flops = FOLDER_NAME + '/flops_autodiff.csv'
ratios = FOLDER_NAME + '/ratios_autodiff.csv'

headers1 = ['input_dim','output_dim','eval','fwd','bwd']
headers2 = ['input_dim','output_dim','fwd','bwd']

with open(flops, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers1)
    writer.writeheader()
file.close()

with open(ratios, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers2)
    writer.writeheader()
file.close()
    
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
        file.close()
        
        with open(ratios, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers2)
            writer.writerow(ratios_dict)
        file.close()
        
        print(f"line in {flops} and {ratios}")
        

print("\n'outerFLOPs.py' run finished!")




