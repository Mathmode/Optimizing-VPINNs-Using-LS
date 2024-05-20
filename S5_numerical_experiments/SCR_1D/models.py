#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Last edited on May, 2024

@author: curiarteb
"""

from config import A,B,N
import os, tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)

keras.utils.set_random_seed(1234)


class u_net(keras.Model):
    def __init__(self, **kwargs):
        super(u_net, self).__init__()
        
        # Hidden layers
        self.layers_list = [keras.layers.Dense(units=N, activation="tanh", use_bias=True) for i in range(2)]
        self.layers_list.append(keras.layers.Dense(units=N, activation="tanh", use_bias=True))
        
        #Last (linear) layer
        self.last_layer = keras.layers.Dense(units=1, activation=None, use_bias=False)

    # Returns the vector output of the neural network
    def call_vect(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)

        x = x*(inputs-A)
        x = x*(inputs-B) # To impose boundary conditions
        return x
    
    # Returns the scalar output of the neural network
    def call(self, inputs):
        out = self.call_vect(inputs)
        return self.last_layer(out)
    
    # Computes the derivative with respect to the input via forward autodiff
    def dfwd(self, inputs):
        x = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tape:
            u = self(x)
        du = tape.jvp(u)
        return du
    
    # Computes the derivative with respect to the input via backward autodiff
    def dbwd(self, inputs):
        x = inputs
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            u = self(x)
        du = tape.gradient(u,x)
        return du
    
    # Computes the derivative with respect to the input via forward autodiff
    # for the vector output
    def dfwd_vect(self, inputs):
        x = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tape:
            u = self.call_vect(x)
        du = tape.jvp(u)
        return du
    
    # Computes the derivative with respect to the input via backward autodiff
    # for the vector output
    def dbwd_vect(self, inputs):
        x = inputs
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            u = self.call_vect(x)
        du = tf.squeeze(tape.batch_jacobian(u,x))
        return du
