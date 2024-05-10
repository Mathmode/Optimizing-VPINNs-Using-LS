#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

import numpy as np
import tensorflow as tf
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) # double precision set to default

# Random seeds
keras.utils.set_random_seed(1234)
np.random.seed(1234)

# External global variables
IMPLEMENTATION = sys.argv[1]
N = int(sys.argv[2])
M = int(sys.argv[3])
K = int(sys.argv[4])
CHECKPOINT = sys.argv[5]

# Booleans for implementation types
# Note: The article DOES NOT cover experiments in strong implementations
if IMPLEMENTATION == "strongfwd":
    STRONG, WEAK, ULTRAWEAK, FORWARD = True, False, False, True
elif IMPLEMENTATION == "strongbwd":
    STRONG, WEAK, ULTRAWEAK, FORWARD = True, False, False, False
elif IMPLEMENTATION == "weakfwd":
    STRONG, WEAK, ULTRAWEAK, FORWARD = False, True, False, True
elif IMPLEMENTATION == "weakbwd":
    STRONG, WEAK, ULTRAWEAK, FORWARD = False, True, False, False
elif IMPLEMENTATION == "ultraweak":
    STRONG, WEAK, ULTRAWEAK, FORWARD = False, False, True, False

# Domain endpoints in 1D
A = np.array(0.)
B = np.pi

class integration_points_and_weights():
    def __init__(self):
        self.grid = np.linspace(A, B, num=K//2+1)
        b_grid = np.expand_dims(self.grid[1:], axis=1)
        a_grid = np.expand_dims(self.grid[:-1], axis=1)
        #ab_halves_grid = (a_grid + b_grid)/2
        self.scale = (b_grid - a_grid)/2
        self.bias = (b_grid + a_grid)/2
        
    # Generates random integration points and weights for exact integration
    # in p=1
    def generate_raw(self):
        x1 = np.expand_dims(np.random.uniform(low=-1, high=1,
                                              size=len(self.grid)-1), axis=1)
        x2 = -x1
        w1 = np.ones_like(x1)
        w2 = np.ones_like(x2)
        return [np.concatenate((x1, x2), axis=1),
                np.concatenate((w1, w2), axis=1)]
    
    # Every time we call it, we produce new integration points and weights
    def __call__(self):
        points, weights = self.generate_raw()
        points = self.scale * points + self.bias
        weights = self.scale * weights
        return [tf.convert_to_tensor(np.expand_dims(points.flatten(), axis=1)),
                tf.convert_to_tensor(np.expand_dims(weights.flatten(), axis=1))]

class u_net(keras.Model):
    def __init__(self, **kwargs):
        super(u_net, self).__init__()
        
        # Hidden layers
        self.layers_list = [keras.layers.Dense(units=2**10, activation="tanh", use_bias=True) for i in range(5)]
        self.layers_list.append(keras.layers.Dense(units=N, activation="tanh", use_bias=True))
        
        #Last (linear) layer
        self.last_layer = keras.layers.Dense(units=1, activation=None, use_bias=False)

    # Returns the vector output of the neural network when fed by the batch of
    # integration points
    def call_vector(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)

        x = x*(inputs-A)
        x = x*(inputs-B) # To impose boundary conditions
        return x
    
    # Returns the scalar output of the neural network when fed by the batch of
    # integration points
    def linear_comb(self, out_vector):
        return self.last_layer(out_vector)
    
    # Returns the scalar output of the network when fed by the batch of
    # integration points
    def call(self, inputs):
        out = self.call_vector(inputs)
        return self.linear_comb(out)

class test_functions(keras.Model):
    def __init__(self, **kwargs):
        super(test_functions,self).__init__()
        
        self.eval = lambda x: keras.ops.concatenate([np.sqrt(2/(B-A))*(B-A)/(np.pi*m)*keras.ops.sin(m*(x-A)/(B-A)) for m in range(1, M+1)], axis=1) # sines normalized in H10
        self.deval = lambda x: keras.ops.concatenate([np.sqrt(2/(B-A))*keras.ops.cos(m * (x-A)/(B-A)) for m in range(1, M+1)], axis=1) # dsines normalized  in H10
        self.ddeval = lambda x: keras.ops.concatenate([-np.sqrt(2/(B-A))*(np.pi*m/(B-A))*keras.ops.sin(m*(x-A)/(B-A)) for m in range(1, M+1)], axis=1) # ddsines normalized in H10
        
    # Evaluation of the test functions
    def call(self, x):
        return self.eval(x)
    
    # Evaluation of the spatial gradient of the test functions
    def d(self, x):
        return self.deval(x)
    
    # Evaluation of the spatial laplacian of the test functions
    def dd(self, x):
        return self.ddeval(x)

# Funtion to terminate the script execution to measure FLOPs until that point
def checkpoint(message):
    if CHECKPOINT == message:
        print("EXIT IN " + message)
        sys.exit()
    else:
        pass
    
class loss_func(keras.Model):
    def __init__(self, net, source, **kwargs):
        super(loss_func, self).__init__()
        
        self.test = test_functions()
        self.f = source
        self.net = net
    
    def call(self, inputs):
        
        x,w = inputs

        #=====================================================================
        # Right-hand side vector construction
        #=====================================================================
        v = self.test(x)
        f = self.f(x)
        RHV = tf.einsum("kr,kr,km->mr", w, f, v)
        
        
        if STRONG:
            #=====================================================================
            # Evaluate the second-order derivative of the network
            #=====================================================================
            if FORWARD:
                with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as touter:
                    with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tinner:
                        u = self.net(x)
                    du = tinner.jvp(u)
                ddu = touter.jvp(du)
                
            else:
                with tf.GradientTape(watch_accessed_variables=False) as touter:
                    touter.watch(x)
                    with tf.GradientTape(watch_accessed_variables=False) as tinner:
                        tinner.watch(x)
                        u = self.net(x)
                    du = tinner.gradient(u,x)
                ddu = touter.gradient(du,x)
            
            #=====================================================================
            #Construct the bilinear-form tensor
            #=====================================================================
            v = self.test(x)
            LHV = tf.einsum("kr,kr,km->mr", w, ddu, v)
            
        elif WEAK:
            #=====================================================================
            # Evaluate the first-order derivative of the network
            #=====================================================================
            if FORWARD:
                with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tape:
                    u = self.net(x)
                du = tape.jvp(u)
                
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(x)
                    u = self.net(x)
                du = tape.gradient(u,x)
        
            #=====================================================================    
            #Construct the bilinear-form tensor
            #=====================================================================
            dv = self.test.d(x)
            LHV = tf.einsum("kr,kr,km->mr", w, du, dv)
                
        elif ULTRAWEAK:
            #=====================================================================    
            # Evaluation of the network
            #=====================================================================
            u = self.net(x)
            
            #=====================================================================
            #Construct the bilinear-form tensor
            #=====================================================================
            ddv = self.test.dd(x)
            LHV = tf.einsum("kr,kr,km->mr", w, u, ddv)
            
        return tf.reduce_sum(tf.square(LHV - RHV))



#*********************************************************************
# Initializations
#*********************************************************************

generate_integration_points_weights = integration_points_and_weights() # Initialize the class that generates integration points and weights
myinputs = generate_integration_points_weights() # Generate integration points and weights
x,w = myinputs
mynet = u_net() # Initialize the u_net class
mynet(x) # Evaluate it for building (initializing variables) 
source_function = lambda x: 1/np.pi*(-4*np.pi*tf.math.cos(2*x)*tf.math.cos(x/np.pi)+(1+4*np.pi**2)*tf.math.sin(2*x)*tf.math.sin(x/np.pi))
myloss = loss_func(mynet, source_function) # Initilize the loss_func class

#=====================================================================
# Initial FLOPs
#=====================================================================
checkpoint("initial")

#=====================================================================
# Baseline FLOPs for the evaluation of the scalar-valued network
#=====================================================================

mynet(x)

checkpoint("net")
    
with tf.GradientTape(watch_accessed_variables=True) as tape:
    
    out = myloss(myinputs)
    checkpoint("loss-trace")
    
#*********************************************************************
# We compute the gradients and apply the GD-based optimizer
#*********************************************************************

dout = tape.gradient(out, myloss.variables)

keras.optimizers.SGD(learning_rate=0.01).apply(dout, myloss.variables)

checkpoint("GD-step")



