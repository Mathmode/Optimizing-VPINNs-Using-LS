# -*- coding: utf-8 -*-
'''
Last edited on May, 2024

@author: curiarteb
'''

import subprocess, os

PYTHON_PATH = "python"
RESULTS_FOLDER = "results"

# Create the results folder if it does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def execute_scripts(script_list):
    for script in script_list:
        try:
            print(f"\nExecuting {script}...")
            with subprocess.Popen([PYTHON_PATH, '-u', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as sp:
                
                # Print stdout and stderr in real-time
                for line in sp.stdout:
                    print(line, end='')
                for line in sp.stderr:
                    print(line, end='')
                    
                sp.wait()
                print(f"\nFinished executing {script}")

        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {script}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while executing {script}: {e}")

script_list = [
    #'S52weak.py', 'S52ultraweak.py',
    #'S53weak.py', 'S53ultraweak.py', 'S53weak_moreiterations.py',
    'S54weak_M32.py', 'S54ultraweak_M32.py',
    'S54weak_M64.py', 'S54ultraweak_M64.py',
    'S54weak_M128.py', 'S54ultraweak_M128.py',
]

# Execute the scripts
execute_scripts(script_list)
