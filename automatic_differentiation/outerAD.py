#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2024

Script to measure FLOPs using 'perf' as underlying tool.

This script is currently optimized to be executed in the David server.

@author: curiarteb
"""

##### Configuration variables #####

#Path to the Python binary that will run 'inner.py'
RUN_PATH = './checkrun.sh'
#Path to the inner script
INNER = 'innerAD.py'
#String to identify the line reporting FLOPS. CPU-dependent!!
FLOPS_ID = 'fp_ret_sse_avx_ops.all'
#Perf command. CPU-dependent!!
COMMAND = f'perf stat -I 1000 -e {FLOPS_ID} {RUN_PATH} {INNER}'

import subprocess, os, csv, shutil
import numpy as np
import pandas as pd

# To give permission to RUN_PATH for execution
subprocess.run(['chmod','a+x', f'{RUN_PATH}'])

def subprocess2measureflops(input_dim, output_dim, checkpoint, error_file="stdout_stderr.txt"):

    sp = subprocess.run([f'{COMMAND} {input_dim} {output_dim} {checkpoint}'], 
                        shell=True, capture_output=True,
                        text=True, check=True)
    
    #print(sp.stdout)
    #print(sp.returncode)
    #print(sp.stderr)  

    # In case the subprocess terminates for some reason different from
    # a regular execution, we terminate and save the 'stdout' and 'stderr'. 
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
    
    # If the execution was correct, we take the FLOPs of the inner execution
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

# Create 'results' folder    
folder_name = "results"
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)
else:
    os.makedirs(folder_name)
    
# Create three csv for different data
flops = folder_name + '/flops_autodiff.csv'
ratios = folder_name + '/ratios_autodiff.csv'

headers1 = ['input_dim','output_dim','eval','fwd','fwd2','bwd','bwd2','fwdbwd','bwdfwd']
headers2 = ['input_dim','output_dim','fwd','fwd2','bwd','bwd2','fwdbwd','bwdfwd']

with open(flops, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers1)
    writer.writeheader()
file.close()

with open(ratios, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers2)
    writer.writeheader()
file.close()
    

sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


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
        
        post = subprocess2measureflops(i, o, "fwd2")
        fwd2 = post - prev
        ratio_fwd2 = fwd2/evaluation
        prev = post
        print("fwd2")
        
        post = subprocess2measureflops(i, o, "bwd")
        bwd = post - prev
        ratio_bwd = bwd/evaluation
        prev = post
        print("bwd")
        
        post = subprocess2measureflops(i, o, "bwd2")
        bwd2 = post - prev
        ratio_bwd2 = bwd2/evaluation
        prev = post
        print("bwd2")
        
        post = subprocess2measureflops(i, o, "fwdbwd")
        fwdbwd = post - prev
        ratio_fwdbwd = fwdbwd/evaluation
        prev = post
        print("fwdbwd")
        
        post = subprocess2measureflops(i, o, "bwdfwd")
        bwdfwd = post - prev
        ratio_bwdfwd = bwdfwd/evaluation
        prev = post
        print("bwdfwd")
        
        flops_dict = {'input_dim':i,
                      'output_dim':o,
                      'eval':"{:.2e}".format(evaluation),
                      'fwd':"{:.2e}".format(fwd),
                      'fwd2':"{:.2e}".format(fwd2),
                      'bwd':"{:.2e}".format(bwd),
                      'bwd2':"{:.2e}".format(bwd2),
                      'fwdbwd':"{:.2e}".format(fwdbwd),
                      'bwdfwd':"{:.2e}".format(bwdfwd)}
        
        ratios_dict = {'input_dim':i,
                      'output_dim':o,
                      'fwd':"{:.2e}".format(ratio_fwd),
                      'fwd2':"{:.2e}".format(ratio_fwd2),
                      'bwd':"{:.2e}".format(ratio_bwd),
                      'bwd2':"{:.2e}".format(ratio_bwd2),
                      'fwdbwd':"{:.2e}".format(ratio_fwdbwd),
                      'bwdfwd':"{:.2e}".format(ratio_bwdfwd)}
        
        # Load data in the csv-s
        with open(flops, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers1)
            writer.writerow(flops_dict)
        file.close()
        
        with open(ratios, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers2)
            writer.writerow(ratios_dict)
        file.close()
        
        print(f"line in {flops} and {ratios}")
        
# Postprocess the generated data for Figures 3 and 4 in the manuscript:

ratios_data = pd.read_csv(ratios)

# For "figure3_data.csv":
figure3_data = ratios_data[['input_dim', 'output_dim', 'fwd', 'bwd']]
figure3_data.to_csv(folder_name + "/figure3_data.csv", index=False)

# For "figure4_data.csv":
figure4_data = ratios_data[ratios_data['input_dim'] == 1][['output_dim', 'fwd', 'fwd2', 'bwd', 'bwd2']]
figure4_data.to_csv(folder_name + "/figure4_data.csv", index=False)

print("\nfigure3_data.csv and figure4_data.csv")
print("\n'outerFLOPs.py' run finished!")




