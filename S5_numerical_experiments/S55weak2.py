# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb
'''
EXPERIMENT_REFERENCE = "S55weak2"
RESULTS_FOLDER = "results"

import numpy as np, os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)

# Random seeds
keras.utils.set_random_seed(1234)
np.random.seed(1234)

from SCR_2D.create_config import create_config

PI = np.pi

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import keras, numpy as np, tensorflow as tf"
EXACT = "lambda x,y : (keras.ops.sin(2*x))*(keras.ops.sin(2*y))"
EXACT_NORM2 = 2 * PI**2
SOURCE = """lambda x,y : 8 * keras.ops.sin(2 * x) * keras.ops.sin(2 * y)"""
A = 0
B = PI
N = 64
M1 = 16
M2 = 16
K1 = 8*M1
K2 = 8*M2
KTEST1 = 2*K1
KTEST2 = 2*K2
IMPLEMENTATION = "weak"
LEARNING_RATE = 10**(-4)
EPOCHS = 1000
XYPLOT = f"[tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num={KTEST1}), axis=1)),tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num={KTEST2}), axis=1))]"
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
                    M1,
                    M2,
                    K1,
                    K2,
                    KTEST1,
                    KTEST2,
                    IMPLEMENTATION,
                    LEARNING_RATE,
                    EPOCHS,
                    XYPLOT,
                    LEGEND_LOCATION]

# Create the 'config.py' script
create_config(global_variables)

# Load the entire script that loads the neural networks, sets the
# training framework and saves the obtained data
with open('SCR_2D/train_and_save.py', 'r') as file:
    script_contents = file.read()

# Execute it
exec(script_contents)