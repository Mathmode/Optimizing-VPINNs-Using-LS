# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@author: Manuela Bastidas
'''

import tensorflow as tf
import numpy as np

import os.path

from SCR.plot_comp import plot_loss 

tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)

# =============================================================================
# The parameters
# =============================================================================

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
regul = 10**(-10)

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
name = 'ResultsP2_lesscomp'
final_directory = os.path.join(current_directory, name)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


# # =============================================================================
# # Saving the solution and plots
# # =============================================================================
   
# np.save(f'{final_directory}/my_history.npy' ,history.history)

# # PLOTS 
# plot_sol(u_model,exact_u,du_exact,nn_last,WEAK,LS,final_directory_fig)

historyWeak = np.load('/Users/manuela/Desktop/UltraweakImplementations/ultra_weak_1D_randomInt/ResultsP2_less_unif/WTrue_LSFalse/my_history.npy',allow_pickle=True).item()
historyUltraweak = np.load('/Users/manuela/Desktop/UltraweakImplementations/ultra_weak_1D_randomInt/ResultsP2_less_unif/WFalse_LSFalse/my_history.npy',allow_pickle=True).item()

plot_loss(historyWeak,historyUltraweak,exact_norm,final_directory)

#plot_loss(u_model,history,nn_last,WEAK,LS[1],exact_norm,final_directory)
   



