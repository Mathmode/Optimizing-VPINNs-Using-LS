#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author: Manuela Bastidas
"""
import tensorflow as tf
import numpy as np

from SCR.callbackPlots import PlotSolutionCallback

# =============================================================================
# Auxiliary function for the loss, validation and error
# =============================================================================

## Loss function for training
def tricky_loss(y_true,y_pred):
    return y_pred[0]

## Loss function for validation
def sqrt_val(y_true,y_pred):
    return tf.sqrt(y_pred[1])

## Error H1 or H01 (if present is in pos -1)
def error(y_true, y_pred):
    return y_pred[-1]


# =============================================================================
# Training function
# =============================================================================

def training(loss_model,learning_rate,iterations,VAL,ERR,u_exact,dux_exact,duy_exact,dir_figs):
    
    # Optimizer (Adam optimizer with a specific learning rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    metrics = []
    callback = [PlotSolutionCallback(u_exact,dux_exact,duy_exact,dir_figs)]
    # If validation is True we also report the loss validation at each iteration
    if VAL: 
        metrics.append(sqrt_val)
        
    # If errror is True we also report the error H_0^1 at each iteration
    if ERR: 
        metrics.append(error)
    
    # Compile the loss model with a custom loss function (tricky_loss)
    loss_model.compile(optimizer=optimizer,loss= tricky_loss, metrics = metrics)
    
    # Train the model using a single training data point ([1.], [1.]) --DUMMY --
    # for a specified number of epochs (iterations)
    history = loss_model.fit(np.array([1.]), np.array([1.]), epochs=iterations,
                             callbacks=[callback])
    
    return history