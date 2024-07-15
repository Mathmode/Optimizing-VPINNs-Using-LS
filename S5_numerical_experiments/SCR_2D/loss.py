#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on June, 2024

@author: curiarteb
"""

from config import K1,K2,KTEST1,KTEST2,SOURCE, EXACT, EXACT_NORM2, IMPLEMENTATION, LEARNING_RATE
from SCR_2D.integration import integration_points_and_weights
from SCR_2D.models import error, test_functions
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) 

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
        self.errorGD = error(netGD)
        
        # A copy of the neural network for the LSGD training
        self.netLSGD = netLSGD
        self.errorLSGD = error(netLSGD)
        
        # Weights for the GD-based optimization for 'netLSGD'
        self.trainable_varsGD = [v.value for v in self.netGD.weights]
        self.trainable_varsLSGD = [v.value for v in self.netLSGD.weights[:-1]]
        # Weights for the LS computations for 'netLSGD'
        self.computable_varsLSGD = self.netLSGD.weights[-1].value
        
        # Data generators:
        self.train_data = integration_points_and_weights(threshold=[K1,K2])
        self.test_data = integration_points_and_weights(threshold=[KTEST1,KTEST2])
        
    def build(self, input_shape):
        super(loss_GDandLSGD, self).build(input_shape)
    
    def construct_LHMandRHV(self, data):
        
        x,y,w = data
        inputs = [x,y]

        # Right-hand side vector construction
        v = self.test(inputs)
        f = self.f(x,y)
        self.l = tf.einsum("kr,kr,km->mr", w, f, v)
        
        # Left-hand side bilinear-form matrix construction
        if IMPLEMENTATION == "weak":
            duX, duY = self.netLSGD.dfwd_vect(inputs)
            dvX, dvY = self.test.gradient(inputs)
            wduXdvX = tf.einsum("kr,kn,km->mn", w, duX, dvX)
            wduYdvY = tf.einsum("kr,kn,km->mn", w, duY, dvY)
            self.B = wduXdvX + wduYdvY
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.netLSGD.call_vect(inputs)
            laplacianv = self.test.laplacian(inputs)
            self.B = tf.einsum("kr,kn,km->mn", w, -u, laplacianv)
            
        return self.B, self.l
    
    # Solve the LS system of linear equations with l2 regularization
    def optimal_computable_vars(self, regularizer = 10**(-8)):
        
        weights_optimal = tf.linalg.lstsq(self.B, self.l, l2_regularizer=regularizer)
        self.computable_varsLSGD.assign(weights_optimal)
        
        return weights_optimal
        
    # Compute || B w - l ||^2 given B, w and l
    def from_LHMandRHV_to_loss(self):
        residual = self.B @ self.net.weights[-1] - self.l
        loss = tf.reduce_sum(residual**2)
        
        return loss
    
    # This is the ordinary DFR loss evaluation that does not pass throughout the matrix construction 
    def call(self, data):
        
        x,y,w = data
        inputs = [x,y]

        v = self.test(inputs)
        f = self.f(x,y)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        if IMPLEMENTATION == "weak":
            duX_GD, duY_GD = self.netGD.dbwd(inputs)
            dvX, dvY = self.test.gradient(inputs)
            wduXdvX_GD = tf.einsum("kr,kr,km->mr", w, duX_GD, dvX)
            wduYdvY_GD = tf.einsum("kr,kr,km->mr", w, duY_GD, dvY)
            LHV_GD = wduXdvX_GD + wduYdvY_GD
            
            duX_LSGD, duY_LSGD = self.netLSGD.dbwd(inputs)
            wduXdvX_LSGD = tf.einsum("kr,kr,km->mr", w, duX_LSGD, dvX)
            wduYdvY_LSGD = tf.einsum("kr,kr,km->mr", w, duY_LSGD, dvY)
            LHV_LSGD = wduXdvX_LSGD + wduYdvY_LSGD
                
        elif IMPLEMENTATION == "ultraweak":
            uGD = self.netGD(inputs)
            uLSGD = self.netLSGD(inputs)
            laplacianv = self.test.laplacian(inputs)
            LHV_GD = tf.einsum("kr,kr,km->mr", w, -uGD, laplacianv)
            LHV_LSGD = tf.einsum("kr,kr,km->mr", w, -uLSGD, laplacianv)
            
        return tf.reduce_sum(tf.square(LHV_GD - RHV)), tf.reduce_sum(tf.square(LHV_LSGD - RHV))
    
    def compile(self):
        super().compile()
        
        self.optimizerGD = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.optimizerLSGD = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    def train_step(self, data):
        
        inputs_train = self.train_data()
        xtrain, ytrain, wtrain = inputs_train
        xtest, ytest, wtest =  self.test_data()
        inputs_test = xtest, ytest
        
        self.construct_LHMandRHV(inputs_train)
        self.optimal_computable_vars()
        

        deGDX, deGDY = self.errorGD.d(inputs_test)
        deLSGDX, deLSGDY = self.errorLSGD.d(inputs_test)
        
        errorGD = tf.reduce_sum(wtest * (tf.square(deGDX)+tf.square(deGDY)))
        errorLSGD = tf.reduce_sum(wtest * (tf.square(deLSGDX)+tf.square(deLSGDY)))
        
        # Optimize netGD and netLSGD
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch([self.trainable_varsGD, self.trainable_varsLSGD])
            lossGD, lossLSGD = self(inputs_train)
        dGD = tape.gradient(lossGD, self.trainable_varsGD)
        dLSGD = tape.gradient(lossLSGD, self.trainable_varsLSGD)
        self.optimizerGD.apply(dGD, self.trainable_varsGD)
        self.optimizerLSGD.apply(dLSGD, self.trainable_varsLSGD)
        
        return {"lossGD": lossGD, "lossLSGD": lossLSGD, "error2GD": errorGD, "error2LSGD": errorLSGD, "rel_errorGD": tf.sqrt(errorGD/EXACT_NORM2), "rel_errorLSGD": tf.sqrt(errorLSGD/EXACT_NORM2)}
