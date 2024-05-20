# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb
'''

import numpy as np
import matplotlib.pyplot as plt
import os
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
PACKAGES = "import keras"
EXACT = "lambda x : keras.ops.sin(40*x)*keras.ops.sin(x/2)"
DEXACT = "lambda x : 40*keras.ops.cos(40*x)*keras.ops.sin(x/2)+1/2*keras.ops.cos(x/2)*keras.ops.sin(40*x)"
SOURCE = "lambda x : -40*keras.ops.cos(40*x)*keras.ops.cos(x/2)+6401/4*keras.ops.sin(40*x)*keras.ops.sin(x/2)"
A = 0
B = PI
N = 128
M = 256
K = 16*M # for 'ultraweak', increase to 32*M
KTEST = 8*K
IMPLEMENTATION = "weak"
SAMPLING = "uniform"
LEARNING_RATE = 10**(-2)

global_variables = [PACKAGES, EXACT, DEXACT, SOURCE, A, B, N, M, K, KTEST, IMPLEMENTATION, SAMPLING, LEARNING_RATE]

# Create the 'config.py' script
create_config(global_variables)

# Load all the functions and classes from SCR_1D AFTER creating the 'config.py' script
from SCR_1D.models import u_net
from SCR_1D.loss import loss_GDandLSGD
from config import SOURCE, EXACT

# Other global variables not in 'config.py'
EPOCHS = 500


x = np.expand_dims(np.linspace(A, B, num=500), axis=1)

netGD, netLSGD = u_net(), u_net()
netGD(x), netLSGD(x)
netLSGD.set_weights(netGD.get_weights())

plt.plot(x,EXACT(x),label="exact",color="black")
plt.plot(x,netLSGD(x),label="LS/Adam",color="C0")
plt.plot(x,netGD(x),label="Adam", color="C1")
plt.legend()
plt.title("Network predictions (before training)")
plt.show()

loss = loss_GDandLSGD(netGD, netLSGD)
loss.compile()
training = loss.fit(x, epochs=EPOCHS, batch_size=K)

loss.construct_LHMandRHV(loss.train_data())
loss.optimal_computable_vars()

plt.plot(x,EXACT(x),label="exact",color="black")
plt.plot(x,netLSGD(x),label="LS/Adam",color="C0")
plt.plot(x,netGD(x),label="Adam", color="C1")
plt.legend()
plt.title("Network predictions (after training)")
plt.show()

plt.plot(x,netLSGD(x)-EXACT(x),label="LS/Adam")
plt.legend()
plt.title("Error")
plt.show()

plt.plot(x,netGD(x)-EXACT(x),label="Adam",color="C1")
plt.legend()
plt.title("Error")
plt.show()

plt.plot(training.history["lossGD"],label="Adam", color="C1")
plt.plot(training.history["lossLSGD"],label="LS/Adam", color="C0")
plt.yscale("log")
plt.legend()
plt.title("Loss history")
plt.show()



    

    