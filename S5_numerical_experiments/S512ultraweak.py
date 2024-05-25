# -*- coding: utf-8 -*-
'''
Last edited on May, 2024

@author: curiarteb
'''

EXPERIMENT_REFERENCE = "S512ultraweak"
RESULTS_FOLDER = "results"

import numpy as np, os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
DEXACT = "lambda x : 40*keras.ops.cos(40*x)*keras.ops.sin(x/2)+1/2*keras.ops.cos(x/2)*keras.ops.sin(40*x)"
SOURCE = "lambda x : -40*keras.ops.cos(40*x)*keras.ops.cos(x/2)+6401/4*keras.ops.sin(40*x)*keras.ops.sin(x/2)"
A = 0
B = PI
N = 64
M = 128
K = 32*M
KTEST = 8*K
IMPLEMENTATION = "ultraweak"
SAMPLING = "uniform"
LEARNING_RATE = 10**(-3)
EPOCHS = 5000
XPLOT = "tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=1000), axis=1))"

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