# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb
'''
EXPERIMENT_REFERENCE = "S55weak"
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
ETA = 1.

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import keras, numpy as np, tensorflow as tf"
EXACT = f"lambda x,y : (keras.ops.sin(2*{ETA}*x))*(keras.ops.sin(2*y))*keras.ops.exp((x+y)/2)"
EXACT_NORM2 =  2 * (-1 + keras.ops.exp(np.pi)) * ((9 + 8 * ETA**2) * (keras.ops.exp(np.pi) - 16 * ETA**2 + 16 * keras.ops.exp(np.pi) * ETA**2) - keras.ops.exp(np.pi) * (9 + 8 * ETA**2) * keras.ops.cos(4 * ETA * np.pi) + 32 * keras.ops.exp(np.pi) * ETA * (-1 + ETA**2) * keras.ops.sin(4 * ETA * np.pi)) / (17 + 272 * ETA**2)
SOURCE = f"""lambda x,y : -2 * keras.ops.exp((x + y) / 2) * keras.ops.cos(2 * y) * keras.ops.sin(2 * {ETA} * x) - \
2 * keras.ops.exp((x + y) / 2) * {ETA} * keras.ops.cos(2 * {ETA} * x) * keras.ops.sin(2 * y) + \
7/2 * keras.ops.exp((x + y) / 2) * keras.ops.sin(2 * {ETA} * x) * keras.ops.sin(2 * y) + \
4 * keras.ops.exp((x + y) / 2) * {ETA}**2 * keras.ops.sin(2 * {ETA} * x) * keras.ops.sin(2 * y)"""
A = 0
B = np.pi
N = 32
M1 = 128
M2 = 128
K1 = 4*M1
K2 = 4*M2
KTEST1 = 4*K1
KTEST2 = 4*K2
IMPLEMENTATION = "weak"
LEARNING_RATE = 10**(-3)
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