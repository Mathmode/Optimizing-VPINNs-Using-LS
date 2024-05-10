#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:56:57 2023

@author: manuela
"""

import tensorflow as tf
import numpy as np

from SCR.integration import integration_points_and_weights

## DFR loss function ( loss into layer )
#Import solution model
class weak_loss(tf.keras.layers.Layer):
    def __init__(self,u_model,n_ptsx,n_ptsy,n_modesx,n_modesy,f,dtype='float64', **kwargs):
        super(weak_loss, self).__init__()
        
        self.u_model = u_model
        self.n_ptsx = n_ptsx
        self.n_ptsy = n_ptsy
        
        # The domain is (0,pi)^2 by default, include as input is the domain change
        a1 = 0
        b1 = np.pi
        a2 = 0
        b2 = np.pi
        
        lenx = b1-a1
        leny = b2-a2
        
        self.num_rules = 100 #negative number indicates a fix rule DST/DCT
        self.select = tf.Variable(0,dtype='int32')
        
        # Generate integration points
        ptsx, Wx = integration_points_and_weights(a1,b1,n_ptsx,self.num_rules)
        ptsy, Wy = integration_points_and_weights(a2,b2,n_ptsy,self.num_rules)
        self.ptsx = tf.constant(ptsx)
        self.ptsy = tf.constant(ptsy)
        
        
        Vx = np.sqrt(2./lenx)
        self.DST_x = tf.constant(np.swapaxes([Vx*np.sin(np.pi*k*(self.ptsx-a1)/lenx)*Wx
                                                for k in range(1,n_modesx+1)], 0, 1))
        
        Vy = np.sqrt(2./leny)
        self.DST_y = tf.constant(np.swapaxes([Vy*np.sin(np.pi*k*(self.ptsy-a2)/leny)*Wy
                                                for k in range(1,n_modesy+1)], 0, 1))
            
        self.DCT_x = tf.constant(np.swapaxes([Vx*(k*np.pi/lenx)*np.cos(np.pi*k*(self.ptsx-a1)/lenx)*Wx
                                             for k in range(1, n_modesx+1)], 0, 1))
        
        self.DCT_y = tf.constant(np.swapaxes([Vy*(k*np.pi/leny)*np.cos(np.pi*k*(self.ptsy-a2)/leny)*Wy
                                             for k in range(1, n_modesy+1)], 0, 1))
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = tf.convert_to_tensor([[ 1/np.pi*((kx**2)/lenx**2 +(ky**2)/leny**2)**-0.5
                                          for kx in range(1,n_modesx+1)] 
                                            for ky in range(1,n_modesy+1)],dtype=dtype)
        self.f = f
        
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.ptsx[self.select]
        y = self.ptsy[self.select]
        
        X,Y = tf.meshgrid(x,y)
        
        Xx = tf.reshape(X,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        Yy = tf.reshape(Y,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(Xx)
            t1.watch(Yy)
            u = self.u_model(tf.stack([Xx,Yy],axis=-1))
        # The derivatives on each direction
        dux = tf.reshape(t1.gradient(u,Xx),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        duy = tf.reshape(t1.gradient(u,Yy),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        del t1
        
        u = tf.reshape(u,[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        f_eval = tf.reshape(self.f(Xx,Yy),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        # Evaluation of the right hand side integral of the weak formultation
        FFT_RHS = tf.einsum("ij,Jj,Ii,IJ->IJ",f_eval,self.DST_x[self.select],
                            self.DST_y[self.select],self.coeffs)
       
        # The result of the LHS integral seen as the dot product of vectors 
        # that translate two the sum of two scalar terms
        FFT_dux = tf.einsum("ij,Jj,Ii,IJ->IJ",dux,self.DCT_x[self.select],
                            self.DST_y[self.select],self.coeffs)
        FFT_duy = tf.einsum("ij,Jj,Ii,IJ->IJ",duy,self.DST_x[self.select],
                            self.DCT_y[self.select],self.coeffs)
        
        return tf.reduce_sum((FFT_dux+FFT_duy+FFT_RHS)**2)
        
    
# DFR loss function ( loss into layer )
#Import solution model
class ultraweak_loss(tf.keras.layers.Layer):
    def __init__(self,u_model,n_ptsx,n_ptsy,n_modesx,n_modesy,f,dtype='float64', **kwargs):
        super(ultraweak_loss, self).__init__()
        
        self.u_model = u_model
        self.n_ptsx = n_ptsx
        self.n_ptsy = n_ptsy
        
        # The domain is (0,pi)^2 by default, include as input is the domain change
        a1 = 0
        b1 = np.pi
        a2 = 0
        b2 = np.pi
        
        lenx = b1-a1
        leny = b2-a2
        
        self.num_rules = 100 #negative number indicates a fix rule DST/DCT
        self.select = tf.Variable(0,dtype='int32')
        
        # Generate integration points
        ptsx, Wx = integration_points_and_weights(a1,b1,n_ptsx,self.num_rules)
        ptsy, Wy = integration_points_and_weights(a2,b2,n_ptsy,self.num_rules)
        self.ptsx = tf.constant(ptsx)
        self.ptsy = tf.constant(ptsy)
        
        
        Vx = np.sqrt(2./lenx)
        self.DST_x = tf.constant(np.swapaxes([Vx*np.sin(np.pi*k*(self.ptsx-a1)/lenx)*Wx
                                                for k in range(1,n_modesx+1)], 0, 1))
        
        Vy = np.sqrt(2./leny)
        self.DST_y = tf.constant(np.swapaxes([Vy*np.sin(np.pi*k*(self.ptsy-a2)/leny)*Wy
                                                for k in range(1,n_modesy+1)], 0, 1))
            
       
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = tf.convert_to_tensor([[((np.pi**2*kx**2)/ lenx**2+(np.pi**2*ky**2)/ leny**2)**-0.5 
                                          for kx in range(1,n_modesx+1)] for ky in range(1,n_modesy+1)],dtype=dtype)
        
        # The coefficients of the laplacian - also seen as the eigenvalues
        self.laplac = tf.convert_to_tensor([[np.pi*((kx**2)/ lenx**2+(ky**2)/ leny**2)**0.5
                                          for kx in range(1,n_modesx+1)] for ky in range(1,n_modesy+1)],dtype=dtype)
        
        self.f = f
        # f_eval = tf.reshape(f(self.X,self.Y),[self.n_ptsy,self.n_ptsx])
        
        # # Evaluation of the right hand side integral of the weak formulation
        # self.FFT_RHS = tf.einsum("ij,Jj,Ii,IJ->IJ",f_eval,self.DST_x,self.DST_y,self.coeffs)
       
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.ptsx[self.select]
        y = self.ptsy[self.select]
        
        X,Y = tf.meshgrid(x,y)
        
        Xx = tf.reshape(X,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        Yy = tf.reshape(Y,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        
        u = self.u_model(tf.stack([Xx,Yy],axis=-1))
        u = tf.reshape(u,[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
    
        f_eval = tf.reshape(self.f(Xx,Yy),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        # Evaluation of the right hand side integral of the weak formultation
        FFT_RHS = tf.einsum("ij,Jj,Ii,IJ->IJ",f_eval,self.DST_x[self.select],
                            self.DST_y[self.select],self.coeffs)
        
        ## Take appropriate transform 
        FT_high = tf.einsum("ij,Jj,Ii,IJ->IJ", u,self.DST_x[self.select],
                            self.DST_y[self.select], self.laplac)
        
        ##Return sum of squares loss 
        return tf.reduce_sum((FT_high + FFT_RHS)**2)

    
## DFR loss function ( loss into layer )
#Import solution model
class error(tf.keras.layers.Layer):
    def __init__(self,u_model,n_pts,dux_exact,duy_exact,dtype='float64', **kwargs):
        super(error, self).__init__()
        
        self.u_model = u_model
        self.n_pts = n_pts
        
        # The domain is (0,pi)^2 by default, include as input is the domain change
        a1 = 0
        b1 = np.pi
        #a2 = 0
        #b2 = np.pi
        
        # Generate integration points
        hi_ptsx = np.linspace(a1,b1, n_pts+1)
        #hi_ptsy = np.linspace(a2,b2, n_pts+1)

        diff = np.abs(hi_ptsx[1:] - hi_ptsx[:-1])
        #diffy = np.abs(hi_ptsy[1:] - hi_ptsy[:-1])
        
        ptsx = tf.constant(hi_ptsx[:-1] + diff / 2)
        #ptsy = tf.constant(hi_ptsy[:-1] + self.diffy / 2)
        
        #puntos entre 0 y pi
        Xx,Yy = tf.meshgrid(ptsx,ptsx)
        self.diffx, self.diffy = tf.meshgrid(diff,diff)
        # puntos en a,b
        
        self.X = tf.reshape(Xx,[n_pts*n_pts])
        self.Y = tf.reshape(Yy,[n_pts*n_pts])
        
        self.duex = dux_exact(Xx,Yy)
        self.duey = duy_exact(Xx,Yy)
        
        self.H01norm = tf.reduce_sum(((self.duex)**2 + (self.duey)**2)*self.diffx*self.diffy)
       
    def call(self,inputs):
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.X)
            t1.watch(self.Y)
            u = self.u_model(tf.stack([self.X,self.Y],axis=-1))
        # Incluiding the error analysis is computationally expensive because 
        # one needs many integration points and derivatives on both directions
        dux = tf.reshape(t1.gradient(u,self.X),[self.n_pts,self.n_pts])
        duy = tf.reshape(t1.gradient(u,self.Y),[self.n_pts,self.n_pts])
        del t1

        # The H01-norm of the error
        H01error = tf.reduce_sum(((self.duex - dux)**2 + (self.duey - duy)**2)*self.diffx*self.diffy)
        
        ##Return the error
        return H01error**0.5 
    
