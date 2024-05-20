# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@author: Manuela Bastidas
'''

import tensorflow as tf
import numpy as np

import time 
import os.path
#import shutil
import matplotlib.pyplot as plt

from SCR.models import make_model, make_loss_model
from SCR.training import training
from SCR.plot import plot_loss, plot_sol, plot_relError


tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)

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

ref_norm = tf.constant(4.4429)

def run_all_plots(nn_last,WEAK,LS,ref_norm):
    
    # =============================================================================
    # The parameters
    # =============================================================================
    
    # Number of hidden layers 
    nl = 5
    # Number of neurons per hidden layer in the neural network
    nn = 40
    #nn_last = 20
    
    # Number of integration points
    n_ptsx = 100
    n_ptsy = 100
    # Number of Fourier modes
    n_modesx = 20
    n_modesy = 20
    # Number of training iterations
    iterations = 10000

    learning_rate = 10**(-3)
    regul = 10**(-10)
    
    # Validation
    VAL = True
    n_pts_val = round(1.17*max(n_ptsx,n_ptsy)) #Number of points to use for validation
    
    # Include the computation of the error
    ERR = True
    n_pts_err = 2*n_pts_val #Number of points in the interval [a,b] to calculate the error
    
    # =============================================================================
    # Building the models (u_nn and loss)
    # =============================================================================
    
    # Initialize the neural network model for the approximate solution
    u_model, u_model_LS = make_model(neurons=nn, n_layers=nl, neurons_last=nn_last, 
                                     train= ~LS[0])
    
    # Big model including the Deep Fourier Regularization (DFR) loss
    loss_model = make_loss_model(u_model,u_model_LS,n_ptsx,n_ptsy,n_modesx,n_modesy,
                                 f_rhs,LS,regul,WEAK,VAL,n_pts_val,ERR,n_pts_err,
                                 dux_exact,duy_exact)
    
    
    # =============================================================================
    # Training 
    # =============================================================================
     
    history = training(loss_model,learning_rate,iterations,VAL,ERR)
    
    current_directory = os.getcwd()
    #init_file = os.path.join(current_directory,'ModelO_DFR.py')
    name = f'Results2D/W_{WEAK}_LS_{LS[0]}{LS[1]}_Nl_{nn_last}'
    final_directory = os.path.join(current_directory, name)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
       
    np.save(f'{final_directory}/my_history.npy' ,history.history)

    plot_sol(u_model,exact_u,dux_exact,duy_exact,nn,WEAK,LS[1],final_directory)

    plot_relError(u_model,ref_norm,history,nn_last,WEAK,LS[1],final_directory)

    plot_loss(u_model,exact_u,history,nn,WEAK,LS[1],final_directory)
    
    #total_param = u_model.count_params()
    #total_param_train = u_model_LS.count_params()
    
    del u_model
    del u_model_LS
    del loss_model
    del history

    #return total_param,total_param_train#,totalTime


def run_all(nn_last,WEAK,LS):
    
    print(f'W_{WEAK}_LS_{LS}_Nl_{nn_last}')
    # =============================================================================
    # The parameters
    # =============================================================================
    
    # Number of hidden layers 
    nl = 5
    # Number of neurons per hidden layer in the neural network
    nn = 40
    #nn_last = 20
    
    # Number of integration points
    n_ptsx = 25
    n_ptsy = 25
    # Number of Fourier modes
    n_modesx = 10
    n_modesy = 10
    # Number of training iterations
    iterations = 500

    learning_rate = 10**(-3)
    regul = 10**(-10)
    
    # Validation
    VAL = True
    n_pts_val = round(1.17*max(n_ptsx,n_ptsy)) #Number of points to use for validation
    
    # Include the computation of the error
    ERR = True
    n_pts_err = 2*n_pts_val #Number of points in the interval [a,b] to calculate the error
    
    # =============================================================================
    # Building the models (u_nn and loss)
    # =============================================================================
    
    # Initialize the neural network model for the approximate solution
    u_model, u_model_LS = make_model(neurons=nn, n_layers=nl, neurons_last=nn_last, 
                                     train= ~LS[0])
    
    # Big model including the Deep Fourier Regularization (DFR) loss
    loss_model = make_loss_model(u_model,u_model_LS,n_ptsx,n_ptsy,n_modesx,n_modesy,
                                 f_rhs,LS,regul,WEAK,VAL,n_pts_val,ERR,n_pts_err,
                                 dux_exact,duy_exact)
    
    # =============================================================================
    # Training 
    # =============================================================================
    
    # The most empirical way of measuring the time 
    start_time = time.perf_counter()
     
    history = training(loss_model,learning_rate,iterations,False,False)
    
    totalTime = time.perf_counter() - start_time
    print('\n Elapsed %.6f sec. \n' % totalTime)
    
    del u_model
    del u_model_LS
    del loss_model
    del history

    return totalTime



# =============================================================================
# Run all networks _ TIME (S) per iter
# =============================================================================

nn_last_v = [20,40,80,160]
nn_last_v0 = [20,40,80,160]

weak_v = [False,True]
LS_v = [True,False]
LS_type = ['analyt','auto']

times = np.zeros((6,4))
a=0
for i,w in enumerate(weak_v):
    for j,ls in enumerate(LS_v):
        if ls:
            for s,tp in enumerate(LS_type):
                if w:
                    for k,nn in enumerate(nn_last_v0):
                        #[p,pls] = run_all_plots(nn,w,[ls,tp])
                        #print(a,i,j,s)
                        #print(a,w,ls,tp)
                        times[a,k] = run_all(nn,w,[ls,tp])
                else:
                    for k,nn in enumerate(nn_last_v):
                        #[p,pls] = run_all_plots(nn,w,[ls,tp])
                        times[a,k] = run_all(nn,w,[ls,tp])
                        #print(a,i,j,s)
                        #print(a,w,ls,tp)
                a=a+1
        else:
            for k,nn in enumerate(nn_last_v):
                #[p,pls] = run_all_plots(nn,w,[ls,'-'])
                #print(a,i,j)
                #print(a,w,ls)
                times[a,k] = run_all(nn,w,[ls,'-'])
            a=a+1
            
    
# =============================================================================
# Plot TIME (S) per iter
# =============================================================================

# Number of training iterations
iterations = 500

fig, ax = plt.subplots()
# Plot the approximate solution obtained from the trained model
plt.plot(nn_last_v0, times[4,:-2]/iterations, color='c', linestyle=':', marker='o')
plt.plot(nn_last_v0, times[3,:-2]/iterations, color='m', linestyle=':', marker='o')
plt.plot(nn_last_v, times[5,:-2]/iterations, color='k', linestyle=':', marker='o')

plt.plot(nn_last_v, times[1,:-2]/iterations, color='c', linestyle='-')
plt.plot(nn_last_v, times[0,:-2]/iterations, color='m', linestyle='-')
plt.plot(nn_last_v, times[2,:-2]/iterations, color='k', linestyle='-')

#ax.set_xscale('log')
ax.set_yscale('log')

plt.legend(['weak_LS-auto','weak_LS','weak',
            'ultraweak_LS-auto','ultraweak_LS','ultraweak'], loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('$N_l$')
plt.ylabel('Time per iteration (s)')

ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
#plt.tight_layout()

plt.savefig('Results2D_time.pdf', format = 'pdf', bbox_inches="tight", transparent = True)
plt.show()


# =============================================================================
# Run two networks - Results and relative error
# =============================================================================

nn_last_plot = [40]

weak_v = [False,True]
LS_v = [True,False]
LS_type = ['analyt','auto']
    
for w in weak_v:
    for ls in LS_v:
        if ls:
            for tp in LS_type:
                for nn in nn_last_plot:
                    #print(w,ls,tp,nn)
                    run_all_plots(nn,w,[ls,tp],ref_norm)
        else:
            for nn in nn_last_plot:
                #print(w,ls,tp,nn)
                run_all_plots(nn,w,[ls,'-'],ref_norm)
                
