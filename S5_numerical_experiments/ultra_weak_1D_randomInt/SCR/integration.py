#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:31:14 2023

@author: Manu
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf


def integration_points_and_weights(a,b,n_pts,n_rules,flag):
        
    if flag == 1:
        grid = np.linspace(a,b,n_pts)
    else:
        
        unif_pts = int(np.round(n_pts/10))

        around_zero = np.exp(-np.linspace(15, 0, num=n_pts-unif_pts)) #From 1 to almost 0 in an exponential distribution
        above_one = np.linspace(1, b, num=unif_pts)
        grid = np.concatenate((around_zero-around_zero[0], above_one), axis=0)
        
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
        
    return points, weights

# # num_rules = 3
# n_rules = 3
# n_pts = 5 
# a = 0 
# b = np.pi 

# grid = np.linspace(a+1,b+1,n_pts)-1

# x1 = np.repeat(np.expand_dims(np.random.uniform(low=0, high=0, size=n_rules),axis=1), n_pts-1,axis=1)
# x2 = np.repeat(np.expand_dims(np.random.uniform(low=0.5, high=0.5, size=n_rules), axis=1), n_pts-1,axis=1)

# w1 = (x2-1/2)/(x2-x1)
# #w2 = 1 - w1

# diff = np.expand_dims(grid[1:], axis=0)-np.expand_dims(grid[:-1], axis=0)
# a_is = np.expand_dims(grid[:-1], axis=0)

# points = np.empty((n_rules,2*(n_pts-1)))
# points[:,0::2] = x1*diff + a_is
# points[:,1::2] = x2*diff + a_is

# weights = np.empty((n_rules,2*(n_pts-1)))
# weights[:,0::2] = w1*diff
# weights[:,1::2] = diff-w1*diff


# fig, ax = plt.subplots()
# plt.scatter(grid,grid,s=1)

# plt.show()
#Plot the approximate solution obtained from the trained model
#plt.vlines(abc,0,num_rules ,color='k')
#plt.plot(pts_uni,pts_uni, color='b', marker='o',linestyle='')
#for i in range(num_rules):
    #plt.plot(pts_non[i],0*pts_non[i]+i, color='b', marker='o',linestyle='')
#plt.plot(pts_non[2],0*pts_non[2]+2., color='r', marker='o',linestyle='')

