# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@author: Manuela Bastidas
'''

import tensorflow as tf
import numpy as np

import time 
import os.path
#import pickle
#import matplotlib.pyplot as plt


from SCR.models import make_model, make_loss_model
from SCR.training import training
from SCR.plot import plot_sol, plot_loss, plot_relError

tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)


# =============================================================================
# The parameters of the run
# =============================================================================

# Exact solution
def exact_u(x,y):
    return tf.sin(2 * x)*tf.sin(2 * y)

# Exact solution
def dux_exact(x,y):
    return 2*tf.cos(2 * x)*tf.sin(2 * y)
def duy_exact(x,y):
    return 2*tf.sin(2 * x)*tf.cos(2 * y)

# PDE RHS 
def f_rhs(x,y):
   return -8*tf.sin(2 * x)*tf.sin(2 * y)

ref_norm = np.sqrt(2*np.pi**2)

# Number of hidden layers 
nl = 3
# Number of neurons per hidden layer in the neural network
nn = 10
nn_last = 20

# Number of integration points
n_ptsx = 50
n_ptsy = 50
# Number of Fourier modes
n_modesx = 20
n_modesy = 20
# Number of training iterations
iterations = 2001

learning_rate = 10**(-3)
regul = 10**(-10)

# Validation
VAL = False
n_pts_val = round(1.17*max(n_ptsx,n_ptsy)) #Number of points to use for validation

# Include the computation of the error
ERR = True
n_pts_err = 2*n_pts_val #Number of points in the interval [a,b] to calculate the error

# Making weak or ultraweak
WEAK = False
    
# Making or not LS
LS = [False, 'analyt']


current_directory = os.getcwd()
#init_file = os.path.join(current_directory,'ModelO_DFR.py')
#name = f'ResultsP1/Nl{nn_last}_W{WEAK}_LS{LS[1]}'
name = f'Results2D/W{WEAK}_LS{LS[0]}'
final_directory = os.path.join(current_directory, name)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

final_directory_fig = os.path.join(current_directory, name,'sol_gif')
if not os.path.exists(final_directory_fig):
   os.makedirs(final_directory_fig)
   

# =============================================================================
# Building the models (u_nn and loss)
# =============================================================================

# Initialize the neural network model for the approximate solution
u_model, u_model_LS = make_model(neurons=nn, n_layers=nl, neurons_last=nn_last, 
                                 train=~LS[0])

# Big model including the Deep Fourier Regularization (DFR) loss
loss_model = make_loss_model(u_model,u_model_LS,n_ptsx,n_ptsy,n_modesx,n_modesy,
                             f_rhs,LS,regul,WEAK,VAL,n_pts_val,ERR,n_pts_err,
                             dux_exact,duy_exact)


# =============================================================================
# Training 
# =============================================================================

# The most empirical way of measuring the time 
start_time = time.perf_counter()
 
history = training(loss_model,learning_rate,iterations,VAL,ERR,
                   exact_u,dux_exact,duy_exact,final_directory_fig)

totalTime = time.perf_counter() - start_time
print('\n Elapsed %.3f sec. \n' % totalTime)

# =============================================================================
# Save and plot the results
# =============================================================================

   
np.save(f'{final_directory}/my_history.npy' ,history.history)

#plot_sol(u_model,exact_u,dux_exact,duy_exact,nn,WEAK,LS,final_directory)

#plot_relError(u_model,ref_norm,history,nn_last,WEAK,LS,final_directory)

plot_loss(u_model,history,nn_last,WEAK,LS,ref_norm,final_directory)
