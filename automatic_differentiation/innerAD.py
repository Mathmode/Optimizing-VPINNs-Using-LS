#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2024

@author: curiarteb
"""

import numpy as np
import jax
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["KERAS_BACKEND"] = "jax"

import keras

# Variable inserted by subprocesses in 'outer.py'
INPUT_DIM = int(sys.argv[1])
OUTPUT_DIM = int(sys.argv[2])
CHECKPOINT = sys.argv[3]

# Funtion to terminate the script execution to measure FLOPs until that point
def checkpoint(message):
    if CHECKPOINT == message:
        print("EXIT IN " + message)
        sys.exit()
    else:
        pass

model = [keras.layers.Input(shape=(INPUT_DIM,))]
[model.append(keras.layers.Dense(units=1024, activation="tanh", use_bias=True)) for i in range(5)]
model.append(keras.layers.Dense(OUTPUT_DIM, activation="tanh"))
model = keras.Sequential(model)
model.compile()

dmodel_fwd = jax.jacfwd(model)
dmodel_bwd = jax.jacrev(model)

ddmodel_fwd2 = jax.jacfwd(dmodel_fwd)
ddmodel_bwdfwd = jax.jacrev(dmodel_fwd)
ddmodel_fwdbwd = jax.jacfwd(dmodel_bwd)
ddmodel_bwd2 = jax.jacrev(dmodel_bwd)


x = np.expand_dims(np.linspace(0,1,INPUT_DIM), axis=0)
model(x)

checkpoint("baseline")

model(x)
checkpoint("eval")
    
dmodel_fwd(x)
checkpoint("fwd")

ddmodel_fwd2(x)
checkpoint("fwd2")
    
dmodel_bwd(x)
checkpoint("bwd")

ddmodel_bwd2(x)
checkpoint("bwd2")

ddmodel_fwdbwd(x)
checkpoint("fwdbwd")

ddmodel_bwdfwd(x)
checkpoint("bwdfwd")




