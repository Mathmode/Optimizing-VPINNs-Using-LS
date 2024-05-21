# -*- coding: utf-8 -*-
'''
Last edited on May, 2024

@author: curiarteb
'''

EXPERIMENT_REFERENCE = "S512weak"
RESULTS_FOLDER = "results"

import numpy as np, tensorflow as tf, os
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
K = 16*M
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
from SCR_1D.save import save_and_plot_net, save_and_plot_loss
from config import SOURCE, EXACT, DEXACT

# Other global variables not in 'config.py'
EPOCHS = 500

x = tf.convert_to_tensor(np.expand_dims(np.linspace(A, B, num=1000), axis=1))

netGD, netLSGD = u_net(), u_net()
netGD(x), netLSGD(x)
netLSGD.set_weights(netGD.get_weights())

# We activate this backend to save all the plt's in 'save_and_plot_net' 
# in a pdf file
from matplotlib.backends.backend_pdf import PdfPages

print()
print("#######################")
print(" PLOTS BEFORE TRAINING ")
print("#######################")
print()
figures1 = save_and_plot_net(x,[netGD,netLSGD],file=False)

print()
print("########################")
print("        TRAINING        ")
print("########################")
print()
loss = loss_GDandLSGD(netGD, netLSGD)
loss.compile()
training = loss.fit(x, epochs=EPOCHS, batch_size=1000)

print()
print("########################")
print("  PLOTS AFTER TRAINING  ")
print("########################")
print()
figures2 = save_and_plot_loss(training.history,file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_training.csv")
figures3 = save_and_plot_net(x,[netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_predictions.csv")

# We save all the figures (figures1, figures2 and figures3) in a pdf.
with PdfPages(f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_figures.pdf") as pdf:
    for fig in figures1 + figures2 + figures3:
        pdf.savefig(fig)