#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

from config import SOURCE, EXACT, IMPLEMENTATION, LEARNING_RATE
from SCR_1D.test import test_functions
from SCR_1D.integration import integration_points_and_weights
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) 

class loss_GD(keras.Model):
    
    # For GD, we only invoke the 'call'
    
    # For LS, we need to invoke:
    # (Step 1) 'construct_LHMandRHV',
    # (Step 2) 'optimal_computable_vars', and
    # (Step 3b) 'call'.
    
    def __init__(self, net, **kwargs):
        super(loss_GD, self).__init__()
        
        self.test = test_functions()
        self.f = SOURCE
        self.exact = EXACT
        
        # Neural network for the conventional GD-based training
        self.net = net
    
        # Weights for the GD-based optimization for 'netLSGD'
        self.trainable_vars = [v.value for v in self.net.weights]
        
        # Data generators:
        self.train_data = integration_points_and_weights(TEST=False)
    
    def compile(self):
        super().compile()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
    # This is the ordinary DFR loss evaluation that does not pass throughout the matrix construction 
    def call(self, inputs):
        
        x,w = inputs

        v = self.test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        if IMPLEMENTATION == "weak":
            du = self.net.dbwd(x)
            dv = self.test.d(x)
            LHV = tf.einsum("kr,kr,km->mr", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.netGD(x)
            ddv = self.test.dd(x)
            LHV = tf.einsum("kr,kr,km->mr", w, -u, ddv)
            
        return tf.reduce_sum(tf.square(LHV - RHV))
    
    def train_step(self, data):
        
        inputs = self.train_data()
        
        # Optimize netGD and netLSGD
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            loss = self(inputs)
        dloss = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply(dloss, self.net.trainable_weights)
        
        return {"loss": loss}
    
class loss_LSGD(keras.Model):
    
    # For GD, we only invoke the 'call'
    
    # For LS, we need to invoke:
    # (Step 1) 'construct_LHMandRHV',
    # (Step 2) 'optimal_computable_vars', and
    # (Step 3b) 'call'.
    
    def __init__(self, net, **kwargs):
        super(loss_LSGD, self).__init__()
        
        self.test = test_functions()
        self.f = SOURCE
        self.exact = EXACT
        
        # Neural network for the hybrid LS/GD training
        self.net = net
    
        # Weights for the GD-based optimization for 'netLSGD'
        self.trainable_vars = [v.value for v in self.net.weights[:-1]]
        # Weights for the LS computations for 'netLSGD'
        self.computable_vars = self.net.weights[-1].value
        
        # Data generators:
        self.train_data = integration_points_and_weights(TEST=False)
        self.test_data = integration_points_and_weights(TEST=True)
    
    def construct_LHMandRHV(self, inputs):
        
        x,w = inputs

        # Right-hand side vector construction
        v = self.test(x)
        f = self.f(x)
        l = tf.einsum("kr,kr,km->mr", w, f, v)
        
        # Left-hand side bilinear-form matrix construction
        if IMPLEMENTATION == "weak":
            du = self.net.dfwd_vect(x)
            dv = self.test.d(x)
            B = tf.einsum("kr,kn,km->mn", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.net.call_vect(x)
            ddv = self.test.dd(x)
            B = tf.einsum("kr,kn,km->mn", w, -u, ddv)
            
        return B, l
    
    # Solve the LS system of linear equations with l2 regularization
    def optimal_computable_vars(self, inputs, regularizer = 10**(-8)):
        
        B, l = self.construct_LHMandRHV(inputs)
        weights_optimal = tf.linalg.lstsq(B, l, l2_regularizer=regularizer)
        self.computable_varsLSGD.assign(weights_optimal)
        
        return weights_optimal

    
    # This is the ordinary DFR loss evaluation that does not pass throughout the matrix construction 
    def call(self, inputs):
        
        x,w = inputs

        v = self.test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        if IMPLEMENTATION == "weak":
            du = self.net.dbwd(x)
            dv = self.test.d(x)
            LHV = tf.einsum("kr,kr,km->mr", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.net(x)
            ddv = self.test.dd(x)
            LHV = tf.einsum("kr,kr,km->mr", w, -u, ddv)
            
        return tf.reduce_sum(tf.square(LHV - RHV))
    
    def compile(self):
        super().compile()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    def train_step(self, data):
        
        inputs = self.train_data()
        
        self.optimal_computable_vars(inputs)
        
        # Optimize weights of the hidden layers of 'net'
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_vars)
            loss = self(inputs)
        dloss = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply(dloss, self.trainable_vars)
        
        return {"loss": loss}

class loss_GDandLSGD(keras.Model):
    
    # For GD, we only invoke the 'call'
    
    # For LS, we need to invoke:
    # (Step 1) 'construct_LHMandRHV',
    # (Step 2) 'optimal_computable_vars', and
    # (Step 3b) 'call'.
    
    def __init__(self, netGD, netLSGD, **kwargs):
        super(loss_GDandLSGD, self).__init__()
        
        self.test = test_functions()
        self.f = SOURCE
        self.exact = EXACT
        
        # First version of the neural network
        self.netGD = netGD
        
        # A copy of the neural network for the LSGD training
        self.netLSGD = netLSGD
        
        # Weights for the GD-based optimization for 'netLSGD'
        self.trainable_varsGD = [v.value for v in self.netGD.weights]
        self.trainable_varsLSGD = [v.value for v in self.netLSGD.weights[:-1]]
        # Weights for the LS computations for 'netLSGD'
        self.computable_varsLSGD = self.netLSGD.weights[-1].value
        
        # Data generators:
        self.train_data = integration_points_and_weights(TEST=False)
        self.test_data = integration_points_and_weights(TEST=True)
    
    def construct_LHMandRHV(self, inputs):
        
        x,w = inputs

        # Right-hand side vector construction
        v = self.test(x)
        f = self.f(x)
        self.l = tf.einsum("kr,kr,km->mr", w, f, v)
        
        # Left-hand side bilinear-form matrix construction
        if IMPLEMENTATION == "weak":
            du = self.netLSGD.dfwd_vect(x)
            dv = self.test.d(x)
            self.B = tf.einsum("kr,kn,km->mn", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.netLSGD.call_vect(x)
            ddv = self.test.dd(x)
            self.B = tf.einsum("kr,kn,km->mn", w, -u, ddv)
            
        return self.B, self.l
    
    # Solve the LS system of linear equations with l2 regularization
    def optimal_computable_vars(self, regularizer = 10**(-8)):
        
        weights_optimal = tf.linalg.lstsq(self.B, self.l, l2_regularizer=regularizer)
        self.computable_varsLSGD.assign(weights_optimal)
        
        return weights_optimal
    
    # This is the ordinary DFR loss evaluation that does not pass throughout the matrix construction 
    def call(self, inputs):
        
        x,w = inputs

        v = self.test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        if IMPLEMENTATION == "weak":
            duGD = self.netGD.dbwd(x)
            duLSGD = self.netLSGD.dbwd(x)
            dv = self.test.d(x)
            LHV_GD = tf.einsum("kr,kr,km->mr", w, duGD, dv)
            LHV_LSGD = tf.einsum("kr,kr,km->mr", w, duLSGD, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            uGD = self.netGD(x)
            uLSGD = self.netLSGD(x)
            ddv = self.test.dd(x)
            LHV_GD = tf.einsum("kr,kr,km->mr", w, -uGD, ddv)
            LHV_LSGD = tf.einsum("kr,kr,km->mr", w, -uLSGD, ddv)
            
        return tf.reduce_sum(tf.square(LHV_GD - RHV)), tf.reduce_sum(tf.square(LHV_LSGD - RHV))
    
    def compile(self):
        super().compile()
        
        self.optimizerGD = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.optimizerLSGD = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    def train_step(self, data):
        
        inputs = self.train_data()
        
        self.construct_LHMandRHV(inputs)
        self.optimal_computable_vars()
        
        # Optimize netGD and netLSGD
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch([self.trainable_varsGD, self.trainable_varsLSGD])
            lossGD, lossLSGD = self(inputs)
        dGD = tape.gradient(lossGD, self.trainable_varsGD)
        dLSGD = tape.gradient(lossLSGD, self.trainable_varsLSGD)
        self.optimizerGD.apply(dGD, self.trainable_varsGD)
        self.optimizerLSGD.apply(dLSGD, self.trainable_varsLSGD)
        
        
        
        return {"lossGD": lossGD, "lossLSGD": lossLSGD}
