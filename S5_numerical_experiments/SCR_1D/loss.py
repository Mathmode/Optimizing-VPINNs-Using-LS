#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

from config import IMPLEMENTATION
from test import test_functions
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
    # Alternatively, one might replace
    # Step 3b with (Step 3a) 'from_LHMandRHV_to_loss'. If so,
    # one should modify the 'train_step' in 'model.py'.
    
    def __init__(self, net, source, **kwargs):
        super(loss_GDandLSGD, self).__init__()
        
        self.test = test_functions()
        self.f = source
        self.net = net
        # Weights for the GD-based optimization
        self.trainable_vars = [v.value for v in self.net.weights[:-1]]
        # Weights for the LS computations
        self.computable_vars = self.net.weights[-1].value
    
    def construct_LHMandRHV(self, inputs):
        
        x,w = inputs

        # Right-hand side vector construction
        v = self.test(x)
        f = self.f(x)
        self.l = tf.einsum("kr,kr,km->mr", w, f, v)
        
        # Left-hand side bilinear-form matrix construction
        if IMPLEMENTATION == "weak":

            with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tape:
                u = self.net.call_vector(x)
            du = tape.jvp(u)
            dv = self.test.d(x)
            self.B = tf.einsum("kr,kn,km->mn", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":

            u = self.net.call_vector(x)
            ddv = self.test.dd(x)
            self.B = tf.einsum("kr,kn,km->mn", w, u, ddv)
            
        return self.B, self.l
    
    # Solve the LS system of linear equations with l2 regularization
    def optimal_computable_vars(self, regularizer = 10**(-8)):
        
        weights_optimal = tf.linalg.lstsq(self.B, self.l, l2_regularizer=regularizer)
        self.computable_vars.assign(weights_optimal)
        
        return weights_optimal
        
    # Compute || B w - l ||^2 given B, w and l
    def from_LHMandRHV_to_loss(self):
        residual = self.B @ self.net.weights[-1] - self.l
        loss = tf.reduce_sum(residual**2)
        
        return loss
    
    # This is the ordinary DFR loss evaluation that does not pass throughout the matrix construction 
    def call(self, inputs):
        
        x,w = inputs

        v = self.test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km->mr", w, f, v)
        
        
        if IMPLEMENTATION == "weak":

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                u = self.net(x)
            du = tape.gradient(u,x)
            dv = self.test.d(x)
            LHV = tf.einsum("kr,kr,km->mr", w, du, dv)
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.net(x)
            ddv = self.test.dd(x)
            LHV = tf.einsum("kr,kr,km->mr", w, u, ddv)
            
        return tf.reduce_sum(tf.square(LHV - RHV))
