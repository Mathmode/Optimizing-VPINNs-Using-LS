# -*- coding: utf-8 -*-
'''
Last edited on May, 2024

@author: curiarteb
'''

EXPERIMENT_REFERENCE = "S53weak_moreiterations"
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

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import keras, numpy as np, tensorflow as tf"
EXACT = "lambda x : keras.ops.sin(40*x)*keras.ops.sin(x/2)"
EXACT_NORM2 = 6401*PI/16
SOURCE = "lambda x : -40*keras.ops.cos(40*x)*keras.ops.cos(x/2)+6401/4*keras.ops.sin(40*x)*keras.ops.sin(x/2)"
A = 0
B = PI
N = 64
M = 128
K = 32*M
KTEST = 8*K
IMPLEMENTATION = "weak"
SAMPLING = "uniform"
LEARNING_RATE = 10**(-3)
EPOCHS = 8000
XPLOT = f"tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num={KTEST}), axis=1))"
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