#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:55:33 2023

@author:  Manuela Bastidas
"""

import tensorflow as tf
import numpy as np

from SCR.loss import weak_loss,ultraweak_loss,error
from SCR.loss_LS import weak_loss_LS,ultraweak_loss_LS
from SCR.loss_autoLS import model_autoLS


# =============================================================================
# Definition of the u model
# =============================================================================

##Define approximate solution,simple achitecture
def make_model(neurons,n_layers,neurons_last,train,activation='tanh',dtype='float64'):
    #init = tf.keras.initializers.RandomNormal(stddev=0.1)
    xvals = tf.keras.layers.Input(shape=(1,),name='x_input',dtype=dtype)
    
    ## ---------------
    #  The dense layers
    ## ---------------
    
    # First layer
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype)(xvals)
    for l in range(n_layers-2):
        # Hidden layers
    	l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype)(l1)
    ## Last layer
    l1 = tf.keras.layers.Dense(neurons_last,activation=activation,dtype=dtype)(l1)
    # l2 = tf.keras.layers.Dense(1,activation=activation,dtype=dtype)(l1)
    
    # The cutoff layer implemented on the last layer (linear cutoff)
    # A cut-off layer to impose the boundary conditions (dirichlet 0)
    l2 = cutoff_layer()([xvals,l1])
    
    u_model_LS = tf.keras.Model(inputs = xvals,outputs = l2,name='u_model_LS')
    
    # Last layer is cutomized
    output = linear_last_layer(neurons_last,activation=activation,train=train,dtype=dtype)(l2)

    # Create the model
    u_model = tf.keras.Model(inputs = xvals,outputs = output,name='u_model')
    
    # Print the information of the model u
    u_model.summary()
    return u_model,u_model_LS



# =============================================================================
# Auxiliary layers: Boundary conditions and linear last layer
# =============================================================================

# The cutoff function applied to the last layer 
class cutoff_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(cutoff_layer,self).__init__()
        
    # The analytical cutoff function to impose 0 dir BC
    def call(self,inputs):
        x,N = inputs
        cut = x*(x-np.pi)
        return tf.einsum("ij,ik->ik",cut,N) 
    

# The last layer (customized)
class linear_last_layer(tf.keras.layers.Layer):
    def __init__(self,nn_last,train,dtype='float64',**kwargs):
        super(linear_last_layer,self).__init__()
        
        # Standard initialization of the weights (TF)
        pweight = tf.random.uniform([nn_last],minval=-(6/(1+nn_last))**0.5,
                                    maxval=(6/(1+nn_last))**0.5,dtype =dtype)
        
        # The weights of the last layer (modified in the LS)
        # TRAIN = True -> NO LS
        # TRAIN = False -> LS
        self.vars = tf.Variable(pweight,trainable=train,dtype=dtype)
    
    def call(self,inputs):
        pweights = self.vars #[:-1]
        #bias = self.vars[-1]
        return tf.einsum("i,ji->j",pweights,inputs)#+bias



# =============================================================================
# Definition of the loss model
# =============================================================================

##Make loss model 
def make_loss_model(u_model,u_model_LS,n_pts,n_modes,f,LS,regul,WEAK,VAL,n_pts_val,
                    ERR,n_pts_err,du_exact,nrules,dtype='float64'):
    
    # Input - dummy
    xvals=tf.keras.layers.Input(shape=(1,),name='x_input',dtype=dtype)
    
    # Using the LS losses (Constructing the matrix by definition)
    # If LS and not automatic, if automatic then the traditional loss and wrapping model
    if LS[0] and LS[1] != 'auto':
        # Using the weak or ultraweak implementation of the loss
        if WEAK:
            loss_l = weak_loss_LS(u_model,u_model_LS,regul,n_pts,
                                  n_modes,f,nrules)(xvals)
        else:
            loss_l = ultraweak_loss_LS(u_model,u_model_LS,regul,n_pts,
                                       n_modes,f,nrules)(xvals)
          
    else:
        # Using the weak or ultraweak implementation of the loss
        if WEAK:
            loss_l = weak_loss(u_model,n_pts,n_modes,f,nrules)(xvals)
        else:
            loss_l = ultraweak_loss(u_model,n_pts,n_modes,f,nrules)(xvals)
    
    output = [loss_l]
    # If validation we have an extra layer for the validation
    if VAL:
        if WEAK:
            loss_val = weak_loss(u_model,n_pts_val,n_modes,f,nrules)(xvals)
        else:
            loss_val = ultraweak_loss(u_model,n_pts_val,n_modes,f,nrules)(xvals)
       
        output.append(loss_val)
        
    # If Error calculation we have an extra layer for the error
    if ERR:
        error0 = error(u_model,n_pts_err,du_exact)(xvals)
        output.append(error0)
    
    # The model that has outputs: loss,loss_val and error
    loss_model = tf.keras.Model(inputs=xvals,outputs=tf.stack(output))
    
    # If least squares we wrap the model in other one
    if LS[0] and LS[1]=='auto':
        min_model = model_autoLS(loss_model,regul=regul)
    else:
        min_model = loss_model
        
    return min_model
