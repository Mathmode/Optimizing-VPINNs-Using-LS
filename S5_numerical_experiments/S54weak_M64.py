# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb
'''
EXPERIMENT_REFERENCE = "S54weak_M64"
RESULTS_FOLDER = "results"

import numpy as np, os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)

# Random seeds
keras.utils.set_random_seed(1234)
np.random.seed(1234)

from SCR_1D.create_config import create_config

PI = np.pi
BETA = 0.7

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import numpy as np, tensorflow as tf"
EXACT = f"lambda x : x**{BETA}*({PI}-x)"
EXACT_NORM2 = (BETA*PI**(1+2*BETA))/(-1+4*BETA**2)
SOURCE = f"lambda x : {BETA}*x**({BETA}-2)*({BETA}*x-{PI}*{BETA}+x+{PI})"
A = 0
B = np.pi
N = 16
M = 64
K = 32*M
KTEST = 8*K
IMPLEMENTATION = "weak"
SAMPLING = "exponential"
LEARNING_RATE = 10**(-3)
EPOCHS = 1000
XPLOT = "tf.convert_to_tensor(np.expand_dims(np.linspace(A+10**(-4), B, num=1000), axis=1))"
LEGEND_LOCATION = "best"

global_variables = [EXPERIMENT_REFERENCE,
                    RESULTS_FOLDER,
                    PACKAGES,
                    EXACT,
                    EXACT_NORM2,
                    SOURCE,
                    A,
                    B,
                    N,
                    M,
                    K,
                    KTEST,
                    IMPLEMENTATION,
                    SAMPLING,
                    LEARNING_RATE,
                    EPOCHS,
                    XPLOT,
                    LEGEND_LOCATION]

# Create the 'config.py' script
create_config(global_variables)

# Load the entire script that loads the neural networks, sets the
# training framework and saves the obtained data
with open('SCR_1D/train_and_save.py', 'r') as file:
    script_contents = file.read()

# Execute it
exec(script_contents)