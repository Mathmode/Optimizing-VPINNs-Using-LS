#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on June, 2024

@author: curiarteb
"""

from config import A,B,K1,K2
import numpy as np, tensorflow as tf

class integration_points_and_weights():
    def __init__(self,threshold=[K1,K2]):
        
        num_subintervalsX = threshold[0]//2 + 1
        num_subintervalsY = threshold[1]//2 + 1
        
        self.gridX = np.linspace(A, B, num=num_subintervalsX)
        self.gridY = np.linspace(A, B, num=num_subintervalsY)
            
        # Remaining elements    
        b_gridX = np.expand_dims(self.gridX[1:], axis=1)
        a_gridX = np.expand_dims(self.gridX[:-1], axis=1)
        b_gridY = np.expand_dims(self.gridY[1:], axis=1)
        a_gridY = np.expand_dims(self.gridY[:-1], axis=1)
        self.scaleX = (b_gridX - a_gridX)/2
        self.scaleY = (b_gridY - a_gridY)/2
        self.biasX = (b_gridX + a_gridX)/2
        self.biasY = (b_gridY + a_gridY)/2
        
    # Generates random integration points and weights for exact integration in p=1
    def generate_raw(self):
        
        x1 = np.expand_dims(np.random.uniform(low=-1, high=1,
                                              size=len(self.gridX)-1), axis=1)
        x2 = -x1
        y1 = np.expand_dims(np.random.uniform(low=-1, high=1,
                                              size=len(self.gridY)-1), axis=1)
        y2 = -y1
        x_points = np.concatenate((x1,x2), axis=1)
        y_points = np.concatenate((y1,y2), axis=1)
        wx_points = np.ones_like(x_points)
        wy_points = np.ones_like(y_points)
        
        return [x_points, y_points, wx_points, wy_points]
    
    # Every time we call it, we produce new integration points and weights
    def __call__(self):
        x, y, wx, wy = self.generate_raw()
        x = self.scaleX * x + self.biasX
        y = self.scaleY * y + self.biasY
        wx = self.scaleX * wx
        wy = self.scaleY * wy
        x_con = np.expand_dims(np.concatenate(x),axis=1)
        y_con = np.expand_dims(np.concatenate(y),axis=1)
        wx_con = np.expand_dims(np.concatenate(wx),axis=1)
        wy_con = np.expand_dims(np.concatenate(wy),axis=1)
        x_rep = np.repeat(x_con, y_con.shape[0], axis=0)
        y_rep = np.tile(y_con, (x_con.shape[0], 1))
        wx_rep = np.repeat(wx_con, wy_con.shape[0], axis=0)
        wy_rep = np.tile(wy_con, (wx_con.shape[0], 1))
        w = wx_rep * wy_rep

        return [tf.convert_to_tensor(x_rep), tf.convert_to_tensor(y_rep), tf.convert_to_tensor(w)]
    

