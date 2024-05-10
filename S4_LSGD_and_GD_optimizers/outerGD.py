#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

This script is currently optimized to be executed in 'David' or 'Goliat' servers.

@author: curiarteb
"""

import sys, subprocess, csv
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
INNER = 'innerGD.py'
# Perf command. CPU-dependent!!
COMMAND = PERF + [RUN_PATH, INNER]
# Results folder
RESULTS_FOLDER = sys.argv[1]
# Error record txt
ERROR_FILE = sys.argv[2]

def subprocess2measureflops(imple, N, M, K, label):
    
    process_command = COMMAND + [imple, str(N), str(M), str(K), label]

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
        # Not sure why output is sent to stderr
        out = sp.stderr.split('\n')
        # Look for the interesting lines
        lines = [l for l in out if FLOPS_ID in l]
        # Add all the contributions in the list of flops
        flops_list = [int(line.split()[1].replace('.', '')) for line in lines]
        flops = sum(flops_list)
        return flops

# Experimentation parameters
imple_list = ['ultraweak','weakfwd','weakbwd']
N_list = [2**k for k in range(1,10)]
    
# Create two csv's for different data
flops = RESULTS_FOLDER + '/flops_GD.csv'
ratios = RESULTS_FOLDER + '/ratios_GD.csv'

headers1 = ['imple','N','M','K','net','loss-trace','GD-step','total']
headers2 = ['imple','N','M','K','C=net/K','loss-trace/KC',
            'GD-step/loss-trace','total/KC']

with open(flops, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers1)
    writer.writeheader()
    
with open(ratios, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers2)
    writer.writeheader()

# Begin experimentation
for i in imple_list:
    for N in N_list:
        # We define M and K dependent of N as follows:
        M=5*N
        K=2*M
        
        # Some printings for monitorization during execution
        print('\n')
        print('*********outerGD.py********')
        print(i)
        print('N  M  K:', N, M, K)
        print('###########FLOPS###########')
        
        initial = subprocess2measureflops(i, N, M, K, 'initial')
        prev = initial
        print('initial:   ', '{:.2e}'.format(initial))
        
        post = subprocess2measureflops(i, N, M, K, 'net')
        net = post - prev
        prev = post
        print('net:       ', '{:.2e}'.format(net))
    
        post = subprocess2measureflops(i, N, M, K, 'loss-trace')
        losstrace = post - prev
        prev = post
        print('loss-trace:', '{:.2e}'.format(losstrace))
        
        post = subprocess2measureflops(i, N, M, K, 'GD-step')
        GDstep = post - prev
        print('GD-step:   ', '{:.2e}'.format(GDstep))
        
        total = losstrace + GDstep
        print('total:     ', '{:.2e}'.format(total))
        
        averages_row = {'imple':i,
                        'N':N,
                        'M':M,
                        'K':K,
                        'net':"{:.2e}".format(net),
                        'loss-trace':"{:.2e}".format(losstrace),
                        'GD-step':"{:.2e}".format(GDstep),
                        'total':"{:.2e}".format(total)}
        
        ratios_row = {'imple':i,
                      'N':N,
                      'M':M,
                      'K':K,
                      'C=net/K':"{:.2e}".format(net/K),
                      'loss-trace/KC':"{:.2e}".format(losstrace/net),
                      'GD-step/loss-trace':"{:.2e}".format(GDstep/losstrace),
                      'total/KC':"{:.2e}".format(total/net)}
        
        # Load data in the csv-s
        with open(flops, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers1)
            writer.writerow(averages_row)
        
        with open(ratios, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers2)
            writer.writerow(ratios_row)
