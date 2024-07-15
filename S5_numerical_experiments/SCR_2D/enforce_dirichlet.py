#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on July, 2024

@author: curiarteb
"""

from config import A,B
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)


class d(keras.Model):
    
    def __init__(self, x0, xf, y0, yf, **kwargs):
        super(d, self).__init__()
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf
        
    def build(self, input_shape):
        super(d, self).build(input_shape)
    
    def call(self, inputs):
        
        x,y = inputs
        out = ((x-self.x0)*(self.yf-self.y0)-(y-self.y0)*(self.xf-self.x0))/(B-A)
        return out
    
class t(keras.Model):
    
    def __init__(self, x0, xf, y0, yf, **kwargs):
        super(t, self).__init__()
        self.xc = (x0+xf)/2
        self.yc = (y0+yf)/2
        
    def build(self, input_shape):
        super(t, self).build(input_shape)
    
    def call(self, inputs):
        
        x,y = inputs
        out = (1/(B-A))*((B-A)**2/4-(x-self.xc)**2-(y-self.yc)**2)
        return out
    
class phi(keras.Model):
    
    def __init__(self, x0, xf, y0, yf, **kwargs):
        super(phi, self).__init__()
        self.d = d(x0, xf, y0, yf)
        self.t = t(x0, xf, y0, yf)
        
    def build(self, input_shape):
        super(phi, self).build(input_shape)
    
    def call(self, inputs):
        out = tf.sqrt((self.d(inputs))**2 + 1/4*(tf.sqrt((self.t(inputs))**2+(self.d(inputs))**4)-self.t(inputs))**2)
        return out
        

class enforce_dirichlet(keras.Model):
    
    def __init__(self, **kwargs):
        super(enforce_dirichlet, self).__init__()
        
        self.phi1 = phi(A,B,A,A)
        self.phi2 = phi(A,A,A,B)
        self.phi3 = phi(B,B,A,B)
        self.phi4 = phi(A,B,B,B)
        
    def build(self, input_shape):
        super(enforce_dirichlet, self).build(input_shape)
    
    def call(self, inputs):
        out = self.phi1(inputs) * self.phi2(inputs) * self.phi3(inputs) * self.phi4(inputs)
        return out
    
  