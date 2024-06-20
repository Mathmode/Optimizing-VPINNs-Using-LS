#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

from config import A,B,N,M,KTEST,SOURCE,EXACT,IMPLEMENTATION
from SCR_1D.integration import integration_points_and_weights
import os, tensorflow as tf, numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)

keras.utils.set_random_seed(1234)


class u_net(keras.Model):
    def __init__(self, **kwargs):
        super(u_net, self).__init__()
        
        # Hidden layers
        self.hidden_layers = [keras.layers.Dense(units=N, activation="tanh", use_bias=True) for i in range(3)]
        
        #Last (linear) layer
        self.linear_layer = keras.layers.Dense(units=1, activation=None, use_bias=False)
        
    def build(self, input_shape):
        super(u_net, self).build(input_shape)

    # Returns the vector output of the neural network
    def call_vect(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            
        # To impose boundary conditions
        x = x*(inputs-A)*(inputs - B)
        
        return x
    
    # Returns the scalar output of the neural network
    def call(self, inputs):
        out = self.call_vect(inputs)
        return self.linear_layer(out)
    
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
    
class test_functions(keras.Model):
    def __init__(self, spectrum_size=M, **kwargs):
        super(test_functions,self).__init__()
        
        self.M = spectrum_size
        self.eval = lambda x: keras.ops.concatenate([np.sqrt(2*(B-A))/(m*np.pi)*keras.ops.sin(m*np.pi*(x-A)/(B-A)) for m in range(1, self.M+1)], axis=1) # sines normalized in H10
        self.deval = lambda x: keras.ops.concatenate([np.sqrt(2/(B-A))*keras.ops.cos(m*np.pi*(x-A)/(B-A)) for m in range(1, self.M+1)], axis=1) # dsines normalized  in H10
        self.ddeval = lambda x: keras.ops.concatenate([-np.sqrt(2/(B-A))*(m*np.pi/(B-A))*keras.ops.sin(m*np.pi*(x-A)/(B-A)) for m in range(1, self.M+1)], axis=1) # ddsines normalized in H10
        
    # Evaluation of the test functions
    def call(self, x):
        return self.eval(x)
    
    # Evaluation of the spatial gradient of the test functions
    def d(self, x):
        return self.deval(x)
    
    # Evaluation of the spatial laplacian of the test functions
    def dd(self, x):
        return self.ddeval(x)
    
    
class residual(keras.Model):
    def __init__(self, net, **kwargs):
        super(residual, self).__init__()
        
        self.net = net
        self.f = SOURCE
        self.test = test_functions()
        self.integration_data = integration_points_and_weights(threshold=KTEST)
        self.spectrum_test = test_functions(spectrum_size=1024)
        self.spectrum_integration_data = integration_points_and_weights(threshold = 32*8*1024)
        
    def build(self, input_shape):
        super(residual, self).build(input_shape)

    
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

        
    def spectrum(self):
                
        x,w = self.spectrum_integration_data()

        v = self.spectrum_test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        du = self.net.dbwd(x)
        dv = self.spectrum_test.d(x)
        LHV = tf.einsum("kr,kr,km->mr", w, du, dv)
                
        
        positive_projection_coeff = tf.square(LHV - RHV)
        
        return np.arange(1,self.spectrum_test.M+1), positive_projection_coeff.numpy().flatten()
 