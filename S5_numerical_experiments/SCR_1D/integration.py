#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

from config import A,B,K,KTEST,SAMPLING
import numpy as np, tensorflow as tf

class integration_points_and_weights():
    def __init__(self,TEST=False):
        
        if not TEST:
            num_subintervals = K//2 + 1
        else:
            num_subintervals = KTEST//2 + 1
        
        # We generate the integration grid
        if SAMPLING == "uniform":
            # Uniform sampling grid
            self.grid = np.linspace(A, B, num=num_subintervals)
            
        elif SAMPLING == "exponential":
            # Exponential sampling grid in (0,1)
            around_zero = np.exp(-np.linspace(15, 0, num=num_subintervals//2))
            # Uniform sampling grid in (1,pi)
            above_one = np.linspace(1, B, num=num_subintervals - num_subintervals//2)
            self. grid = np.concatenate((np.zeros(1), around_zero[:-1], above_one), axis=0)
            
        # Remaining elements    
        b_grid = np.expand_dims(self.grid[1:], axis=1)
        a_grid = np.expand_dims(self.grid[:-1], axis=1)
        self.scale = (b_grid - a_grid)/2
        self.bias = (b_grid + a_grid)/2
        
    # Generates random integration points and weights for exact integration in p=1
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

