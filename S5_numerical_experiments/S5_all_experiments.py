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
    # 'S511weak.py', 'S511ultraweak.py',
    # 'S512weak.py', 'S512ultraweak.py', 
    'S521weak.py', 'S521ultraweak.py',
    'S522weak.py', 'S522ultraweak.py', 
    'S523weak.py', 'S523ultraweak.py'
]

# Execute the scripts
execute_scripts(script_list)
