#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Last edited on May, 2024

@author: curiarteb
"""

from config import A,B,N,SOURCE,EXACT,IMPLEMENTATION
from SCR_1D.test import test_functions
from SCR_1D.integration import integration_points_and_weights
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
    
    # Computes the 2nd order derivative with respect to the input 
    # via forward autodiff for the vector output
    def ddfwd(self, inputs):
        x = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as touter:
            with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tinner:
                u = self(x)
            du = tinner.jvp(u)
        ddu = touter.jvp(du)
        return ddu
    
    # Computes the 2nd order derivative with respect to the input 
    # via backward autodiff for the vector output
    def ddbwd(self, inputs):
        x = inputs
        with tf.GradientTape(watch_accessed_variables=False) as touter:
            touter.watch(x)
            with tf.GradientTape(watch_accessed_variables=False) as tinner:
                tinner.watch(x)
                u = self(x)
            du = tinner.gradient(u,x)
        ddu = touter.gradient(du,x)
        return ddu
    
class error(keras.Model):
    def __init__(self, net, **kwargs):
        super(error, self).__init__()
        
        self.net = net
        self.error = EXACT
    
    # Returns the scalar output of the error function
    def call(self, inputs):
        u = self.net(inputs)
        error = self.error(inputs)
        return u - error
    
    def d(self, inputs):
        x = inputs
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            e = self(x)
        de = tape.gradient(e,x)
        return de
    
class residual(keras.Model):
    def __init__(self, net, **kwargs):
        super(residual, self).__init__()
        
        self.net = net
        self.f = SOURCE
        self.test = test_functions()
        self.integration_data = integration_points_and_weights(TEST=False)

    
    def call(self, inputs):
        
        x,w = self.integration_data()

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
        
        projection_coeff = LHV - RHV
        
        v_eval = self.test(inputs)
        
        projection = tf.einsum("mr,km->kr",projection_coeff,v_eval)
        
        return projection
    
    def d(self, inputs):
        x = inputs
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            r = self(x)
        dr = tape.gradient(r,x)
        return dr

        
 