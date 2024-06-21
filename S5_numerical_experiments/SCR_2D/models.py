#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on June, 2024

@author: curiarteb
"""

from config import A,B,N,M1,M2,SOURCE,EXACT,IMPLEMENTATION
from SCR_2D.integration import integration_points_and_weights
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
        out = keras.ops.concatenate(inputs, axis=1)
        for layer in self.hidden_layers:
            out = layer(out)
            
        # To impose boundary conditions
        out = out*(inputs[0]-A)*(inputs[1]-A)*(inputs[0]-B)*(inputs[1]-B)
        
        return out
    
    # Returns the scalar output of the neural network
    def call(self, inputs):
        out = self.call_vect(inputs)
        return self.linear_layer(out)
    
    # Computes the derivative with respect to the input via forward autodiff
    def dfwd(self, inputs):
        x,y = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tapeX, tf.autodiff.ForwardAccumulator(primals=y, tangents=tf.ones_like(y)) as tapeY:
            u = self(inputs)
        duX = tapeX.jvp(u)
        duY = tapeY.jvp(u)
        return [duX,duY]
    
    # Computes the derivative with respect to the input via backward autodiff
    def dbwd(self, inputs):
        x,y = inputs
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            u = self(inputs)
        du = tape.gradient(u,inputs)
        return du
    
    # Computes the derivative with respect to the input via forward autodiff
    # for the vector output
    def dfwd_vect(self, inputs):
        x,y = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tapeX, tf.autodiff.ForwardAccumulator(primals=y, tangents=tf.ones_like(y)) as tapeY:
            u = self.call_vect(inputs)
        duX = tapeX.jvp(u)
        duY = tapeY.jvp(u)
        return [duX, duY]
    
    # Computes the 2nd order derivative with respect to the input 
    # via backward autodiff for the vector output
    def ddbwd(self, inputs):
        x,y = inputs
        with tf.GradientTape(watch_accessed_variables=False) as touter:
            touter.watch(inputs)
            with tf.GradientTape(watch_accessed_variables=False) as tinner:
                tinner.watch(inputs)
                u = self(inputs)
            du = tinner.gradient(u,inputs)
        ddu = touter.gradient(du,inputs)
        return ddu
    
class error(keras.Model):
    def __init__(self, net, **kwargs):
        super(error, self).__init__()
        
        self.net = net
        self.exact = EXACT
    
    # Returns the scalar output of the error function
    def call(self, inputs):
        x,y=inputs
        u = self.net(inputs)
        return u - self.exact(x,y)
    
    def d(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            e = self(inputs)
        de = tape.gradient(e,inputs)
        return de
    
class test_functions(keras.Model):
    def __init__(self, spectrum_size=[M1,M2], **kwargs):
        super(test_functions,self).__init__()
        
        self.M1,self.M2 = spectrum_size
        self.eval = lambda x,y: keras.ops.concatenate([2/(np.pi**2*np.sqrt(m1**2+m2**2))*keras.ops.sin(m1*np.pi*(x-A)/(B-A))*keras.ops.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # sines normalized in H10
        self.devalX = lambda x,y: keras.ops.concatenate([2*m1/(np.pi*np.sqrt(m1**2+m2**2))*keras.ops.cos(m1*np.pi*(x-A)/(B-A))*keras.ops.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesX normalized  in H10
        self.devalY = lambda x,y: keras.ops.concatenate([2*m2/(np.pi*np.sqrt(m1**2+m2**2))*keras.ops.sin(m1*np.pi*(x-A)/(B-A))*keras.ops.cos(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        self.ddevalXX = lambda x,y: keras.ops.concatenate([-2*m1**2/(np.pi*np.sqrt(m1**2+m2**2))*keras.ops.sin(m1*np.pi*(x-A)/(B-A))*keras.ops.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        self.ddevalYY = lambda x,y: keras.ops.concatenate([-2*m2**2/(np.pi*np.sqrt(m1**2+m2**2))*keras.ops.sin(m1*np.pi*(x-A)/(B-A))*keras.ops.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        
    # Evaluation of the test functions
    def call(self, inputs):
        x,y=inputs
        return self.eval(x,y)
    
    # Evaluation of the spatial gradient of the test functions
    def gradient(self, inputs):
        x,y=inputs
        return [self.devalX(x,y), self.devalY(x,y)]
    
    # Evaluation of the spatial laplacian of the test functions
    def laplacian(self, inputs):
        x,y=inputs
        return self.ddevalXX(x,y) + self.ddevalYY(x,y)
    
    
class residual(keras.Model):
    
    def __init__(self, net, **kwargs):
        super(residual, self).__init__()
        
        self.net = net
        self.f = SOURCE
        self.test = test_functions()
        self.spectrum_test = test_functions(spectrum_size=[64,64])
        self.spectrum_integration_data = integration_points_and_weights(threshold = [4*64,4*64])
        
    def build(self, input_shape):
        super(residual, self).build(input_shape)

    
    def call(self, inputs):
        
        x,y,w = inputs

        v = self.test([x,y])
        f = self.f(x,y)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        if IMPLEMENTATION == "weak":
            duX, duY = self.net.dbwd([x,y])
            dvX, dvY = self.test.gradient([x,y])
            wduXdvX = tf.einsum("kr,kr,km->mr", w, duX, dvX)
            wduYdvY = tf.einsum("kr,kr,km->mr", w, duY, dvY)
            LHV = wduXdvX + wduYdvY
                
        elif IMPLEMENTATION == "ultraweak":
            u = self.net([x,y])
            laplacianv = self.test.laplacian([x,y])
            LHV = tf.einsum("kr,kr,km->mr", w, -u, laplacianv)
        
        projection_coeff = LHV - RHV
        
        projection = tf.einsum("mr,km->kr",projection_coeff,v)
        
        return projection
    
    def d(self, inputs):
        
        x,y,w=inputs
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([x,y])
            r = self(inputs)
        dr = tape.gradient(r,[x,y])
        return dr

        
    def spectrum(self):
                
        x,y,w = self.spectrum_integration_data()
        inputs = [x,y]

        v = self.spectrum_test(inputs)
        f = self.f(x,y)
        RHV = tf.einsum("kr,kr,km -> mr", w, f, v)
        
        
        duX, duY = self.net.dbwd(inputs)
        dvX, dvY = self.spectrum_test.gradient(inputs)
        wduXdvX = tf.einsum("kr,kr,km->mr", w, duX, dvX)
        wduYdvY = tf.einsum("kr,kr,km->mr", w, duY, dvY)
        LHV = wduXdvX + wduYdvY
        
        positive_projection_coeff = tf.square(LHV - RHV)
        
        modesX = np.array([m1 for m1 in range(1, 65) for m2 in range(1, 65)])
        modesY = np.array([m2 for m1 in range(1, 65) for m2 in range(1, 65)])
        
        return modesX, modesY, positive_projection_coeff.numpy().flatten()
    
    def spectrum_in_out(self, modesX, modesY, coeff):
        

        in_mask = (modesX <= M1) & (modesY <= M2)
        out_mask = (modesX > M1) | (modesY > M2)
        
        
        value_in = np.sum(coeff[in_mask])
        value_out = np.sum(coeff[out_mask])
        value_total = value_in + value_out
        
        return value_in, value_out, value_total

        
 