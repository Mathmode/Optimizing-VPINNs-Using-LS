# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb
'''
EXPERIMENT_REFERENCE = "S523weak"
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

PI = np.pi
BETA = 0.6

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import numpy as np, tensorflow as tf"
EXACT = f"lambda x : x**{BETA}*({PI}-x)"
DEXACT = f"lambda x : -x**({BETA}-1)*({BETA}*(x-{PI})+x)"
SOURCE = f"lambda x : {BETA}*x**({BETA}-2)*({BETA}*x-{PI}*{BETA}+x+{PI})"
A = 0
B = np.pi
N = 64
M = 128
K = 32*M
KTEST = 8*K
IMPLEMENTATION = "weak"
SAMPLING = "exponential"
LEARNING_RATE = 10**(-3)
EPOCHS = 5000
XPLOT = "tf.convert_to_tensor(np.expand_dims(np.linspace(A+10**(-4), B, num=1000), axis=1))"

global_variables = [EXPERIMENT_REFERENCE,
                    RESULTS_FOLDER,
                    PACKAGES,
                    EXACT,
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
                    XPLOT]

# Create the 'config.py' script
create_config(global_variables)

# Load the entire script that loads the neural networks, sets the
# training framework and saves the obtained data
with open('SCR_1D/train_and_save.py', 'r') as file:
    script_contents = file.read()

# Execute it
exec(script_contents)