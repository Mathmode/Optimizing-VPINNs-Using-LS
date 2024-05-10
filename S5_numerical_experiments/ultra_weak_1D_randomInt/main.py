# -*- coding: utf-8 -*-
'''
Created on May, 2024

@author: curiarteb, Manuela Bastidas
'''

import tensorflow as tf
import numpy as np

import time 
import os.path
#import shutil

from SCR.models import make_model, make_loss_model
from SCR.training import training
from SCR.plot import plot_sol, plot_loss #, plot_relError

# #Uncoment this line to see where is running each operation in your code
# #tf.debugging.set_log_device_placement(True) 

# gpus = tf.config.list_physical_devices('GPU')
# #print('\n \n \n Avaliable GPUs : ', gpus)
# gpusi = 0

# if gpus:
#     try:
#         tf.config.set_visible_devices(gpus[gpusi],'GPU')
#         tf.config.set_logical_device_configuration(gpus[gpusi],
#                 [tf.config.LogicalDeviceConfiguration(memory_limit=32768)])
#         print('\n \n \n logical GPUs : ',tf.config.list_logical_devices('GPU'))
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# with tf.device('/device:GPU:0'): # or with tf.device('/device:CPU'):

    
tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)

# =============================================================================
# The parameters
# =============================================================================

# PROBLEM 1: SMOOTH SOLUTION

# # Exact solution
# def exact_u(x):
#     return tf.sin(2*x)*tf.sin(x/np.pi)

# # Exact solution
# def du_exact(x):
#     #return 2*tf.cos(2 * x)
#     return tf.sin(2*x)*(1/np.pi)*tf.cos(x/np.pi) + tf.sin(x/np.pi)*2*tf.cos(2*x)

# # PDE RHS 
# def f_rhs(x):
#     #return -4*tf.sin(2 * x)
#     return -tf.sin(2*x)*(1/np.pi**2)*tf.sin(x/np.pi) + (1/np.pi)*tf.cos(x/np.pi)*2*tf.cos(2*x) - tf.sin(x/np.pi)*4*tf.sin(2*x) + 2*1/np.pi*tf.cos(x/np.pi)*tf.cos(2*x)

# exact_norm = np.sqrt(1/(4*np.pi) + np.pi - (1/2)*np.pi*tf.sin(2.))

# PROBLEM 2: Singular solution

alpha = 0.7
# Exact solution
def exact_u(x):
    return x**alpha*(np.pi-x)

# Exact solution
def du_exact(x):
    return -x**(alpha-1)*(x+alpha*(x-np.pi))

# PDE RHS 
def f_rhs(x):
    return -(alpha*x**(-2 + alpha)*(np.pi - alpha*np.pi + x + alpha*x))

exact_norm = np.sqrt(11.3759)

# Number of hidden layers 
nl = 3
# Number of neurons per hidden layer in the neural network
nn = 10
nn_last = 20

# Number of integration points
n_pts = 200
# Number of Fourier modes
n_modes = 100
# Number of training iterations
iterations = 1000

nrules = 1000

learning_rate = 10**(-4)
# La regularización funciona diferente en LS auto (precondicionador identidad) y LS analitico
regul = 10**(-5)

# Validation
VAL = False
n_pts_val = round(1.17*n_pts) #Number of points to use for validation

# Include the computation of the error
ERR = True
n_pts_err = 1000 #Number of points in the interval [a,b] to calculate the error

# Making weak or ultraweak
WEAK = False
    
# Making or not LS:
    # Options : True/False and Automatic using autodiff (auto) 
    # Possible error from diff combinations!
LS = [True, 'no']


current_directory = os.getcwd()
#init_file = os.path.join(current_directory,'ModelO_DFR.py')
#name = f'ResultsP1/Nl{nn_last}_W{WEAK}_LS{LS[1]}'
name = f'ResultsP2_less/W{WEAK}_LS{LS[0]}'
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
                                 train= ~LS[0])

# Big model including the Deep Fourier Regularization (DFR) loss
loss_model = make_loss_model(u_model,u_model_LS,n_pts,n_modes,f_rhs,LS,regul,
                             WEAK,VAL,n_pts_val,ERR,n_pts_err,du_exact,nrules)


# =============================================================================
# Training 
# =============================================================================

# The most empirical way of measuring the time 
start_time = time.perf_counter()
 
history = training(loss_model,learning_rate,iterations,VAL,ERR,exact_u,du_exact,
                   final_directory_fig)


# totalTime = time.perf_counter() - start_time
# print('\n Elapsed %.6f sec. \n' % totalTime)
# totalTime_iter = totalTime/iterations
# print('\n Elapsed %.6f sec. \n' % totalTime_iter)


# =============================================================================
# Saving the solution and plots
# =============================================================================
   
np.save(f'{final_directory}/my_history.npy' ,history.history)

# PLOTS 
plot_sol(u_model,exact_u,du_exact,nn_last,WEAK,LS,final_directory_fig)

plot_loss(u_model,history,nn_last,WEAK,LS[1],exact_norm,final_directory)

#plot_loss(u_model,history,nn_last,WEAK,LS[1],exact_norm,final_directory)
   



