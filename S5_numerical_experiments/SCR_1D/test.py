#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

from SCR_1d.config import A,B,M
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) 


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