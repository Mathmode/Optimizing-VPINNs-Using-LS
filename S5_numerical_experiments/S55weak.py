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
ETA = 1

# Intermediate expressions for 'EXACT_NORM2'

e_pi = np.exp(PI)

term1 = 1 / (1105 * (1 + 80 * ETA**2 + 1024 * ETA**4))
term2 = 16 * (e_pi - 1)

cos_4eta_pi = np.cos(4 * ETA * PI)
cos_8eta_pi = np.cos(8 * ETA * PI)
sin_4eta_pi = np.sin(4 * ETA * PI)
sin_8eta_pi = np.sin(8 * ETA * PI)

expr = (51 * e_pi +
        4128 * e_pi * ETA**2 -
        50688 * ETA**4 +
        56064 * e_pi * ETA**4 -
        49152 * ETA**6 +
        49152 * e_pi * ETA**6 -
        4 * e_pi * (17 + 1112 * ETA**2 + 1536 * ETA**4) * cos_4eta_pi +
        e_pi * (17 + 320 * ETA**2 + 768 * ETA**4) * cos_8eta_pi -
        248 * e_pi * ETA * sin_4eta_pi -
        15872 * e_pi * ETA**3 * sin_4eta_pi +
        124 * e_pi * ETA * sin_8eta_pi +
        1600 * e_pi * ETA**3 * sin_8eta_pi -
        6144 * e_pi * ETA**5 * sin_8eta_pi)

# Define the values you want to assign to the global variables via 'config.py'
PACKAGES = "import keras, numpy as np, tensorflow as tf"
EXACT = f"lambda x,y : (keras.ops.sin(2*{ETA}*x))**2*(keras.ops.sin(2*y))**2*keras.ops.exp((x+y)/2)"
EXACT_NORM2 = term1 * term2 * expr
SOURCE = f"""lambda x,y : -8 * keras.ops.exp((x + y) / 2) * keras.ops.cos(2 * y) ** 2 * keras.ops.sin(2 * {ETA} * x) ** 2 - \
4 * keras.ops.exp((x + y) / 2) * keras.ops.cos(2 * y) * keras.ops.sin(2 * {ETA} * x) ** 2 * keras.ops.sin(2 * y) - \
8 * keras.ops.exp((x + y) / 2) * {ETA} ** 2 * keras.ops.cos(2 * {ETA} * x) ** 2 * keras.ops.sin(2 * y) ** 2 - \
4 * keras.ops.exp((x + y) / 2) * {ETA} * keras.ops.cos(2 * {ETA} * x) * keras.ops.sin(2 * {ETA} * x) * keras.ops.sin(2 * y) ** 2 + \
15 / 2 * keras.ops.exp((x + y) / 2) * keras.ops.sin(2 * {ETA} * x) ** 2 * keras.ops.sin(2 * y) ** 2 + \
8 * keras.ops.exp((x + y) / 2) * {ETA} ** 2 * keras.ops.sin(2 * {ETA} * x) ** 2 * keras.ops.sin(2 * y) ** 2"""
A = 0
B = np.pi
N = 32
M1 = 16
M2 = 16
K1 = 4*M1
K2 = 4*M2
KTEST1 = 4*K1
KTEST2 = 4*K2
IMPLEMENTATION = "weak"
LEARNING_RATE = 10**(-3)
EPOCHS = 200
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