#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:31:14 2023

@author: Manu
"""

import numpy as np

def integration_points_and_weights(a,b,n_pts,n_rules):
    
    if n_rules > 0:
        
        grid = np.linspace(a,b,n_pts)
        
        x1 = np.repeat(np.expand_dims(np.random.uniform(low=0., high=0.5, size=n_rules),axis=1), n_pts-1,axis=1)
        x2 = np.repeat(np.expand_dims(np.random.uniform(low=0.5, high=1, size=n_rules), axis=1), n_pts-1,axis=1)
    
        w1 = (x2-1/2)/(x2-x1)
        #w2 = 1 - w1
    
        diff = np.expand_dims(grid[1:], axis=0)-np.expand_dims(grid[:-1], axis=0)
        a_is = np.expand_dims(grid[:-1], axis=0)
    
        points = np.empty((n_rules,2*(n_pts-1)))
        points[:,0::2] = x1*diff + a_is
        points[:,1::2] = x2*diff + a_is
        
        weights = np.empty((n_rules,2*(n_pts-1)))
        weights[:,0::2] = w1*diff
        weights[:,1::2] = diff-w1*diff
        
    else:
        grid = np.linspace(a,b,n_pts)
        weights = np.expand_dims(grid[1:], axis=0)-np.expand_dims(grid[:-1], axis=0)
        points = grid[:-1]+weights/2
        
    return points, weights
