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
class weak_loss_LS(tf.keras.layers.Layer):
    def __init__(self,u_model,u_model_LS,n_ptsx,n_ptsy,n_modesx,n_modesy,f,
                 regul,dtype='float64', **kwargs):
        super(weak_loss_LS, self).__init__()
        
        self.u_model = u_model
        self.u_model_LS = u_model_LS
        
        # The regularization parameter
        self.regul = regul
        
        # Number of neurons 
        self.nn = self.u_model.layers[-2].output_shape[1]
        
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
        
        self.coeffs = tf.convert_to_tensor([[ 1/np.pi*((kx**2)/lenx**2 +(ky**2)/leny**2)**-0.5
                                          for kx in range(1,n_modesx+1)] 
                                            for ky in range(1,n_modesy+1)],dtype=dtype)
        self.f = f
        
        self.n_modesx = n_modesx
        self.n_modesy = n_modesy
        
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.ptsx[self.select]
        y = self.ptsy[self.select]
        
        X,Y = tf.meshgrid(x,y)
        
        Xx = tf.reshape(X,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        Yy = tf.reshape(Y,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        
        with tf.autodiff.ForwardAccumulator(primals = Xx,
                                            tangents = tf.ones_like(Xx)) as t1:
            with tf.autodiff.ForwardAccumulator(primals = Yy,
                                                tangents = tf.ones_like(Yy)) as t2:
                u = tf.unstack(self.u_model_LS(tf.stack([Xx,Yy],axis=-1)),axis=-1)
        
        dux = t1.jvp(u)
        duy = t2.jvp(u)
        del t1, t2 
        
        #u = tf.reshape(uY,[self.nn,self.n_ptsy,self.n_ptsx])
        dux = tf.reshape(dux,[self.nn,2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        duy = tf.reshape(duy,[self.nn,2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        FFT_dux = tf.einsum("kij,Jj,Ii,IJ->IJk",dux,self.DCT_x[self.select],
                            self.DST_y[self.select],self.coeffs)
        
        FFT_duy = tf.einsum("kij,Jj,Ii,IJ->IJk",duy,self.DST_x[self.select],
                            self.DCT_y[self.select],self.coeffs)
        
        f_eval = tf.reshape(self.f(Xx,Yy),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        # Evaluation of the right hand side integral of the weak formulation
        vecB = tf.einsum("ij,Jj,Ii,IJ->IJ",f_eval,self.DST_x[self.select],
                              self.DST_y[self.select],self.coeffs)
       
        # The "traditional" LHS of the weak/ultraweak formulation (Nneurons x Nmodesx*Nmodesy)
        #mat_A = tf.squeeze(tf.reshape(FFT_dux,(self.nn,self.n_modesx*self.n_modesy))+
                           #tf.reshape(FFT_duy,(self.nn,self.n_modesx*self.n_modesy)))
        
        mat_A = tf.reshape(FFT_dux+FFT_duy,(self.n_modesx*self.n_modesy,self.nn))
        aux = tf.reshape(vecB,(self.n_modesx*self.n_modesy,1))
        
        # The LS solution: The regularization is very important/sensitive here!!
        solution_w0 = tf.squeeze(tf.linalg.lstsq(mat_A,-aux,l2_regularizer=self.regul))
         
        # Assign the weights
        self.u_model.layers[-1].vars.assign(solution_w0)
        
        # Compute the LHS
        FT_high = tf.einsum("ik,k->i",mat_A,solution_w0)
        
        # The value of the loss function
        return tf.reduce_sum((FT_high+tf.reshape(vecB,(1,self.n_modesx*self.n_modesy)))**2)
        
    
# DFR loss function ( loss into layer )
#Import solution model
class ultraweak_loss_LS(tf.keras.layers.Layer):
    def __init__(self,u_model,u_model_LS,n_ptsx,n_ptsy,n_modesx,n_modesy,f,regul,
                 dtype='float64', **kwargs):
        super(ultraweak_loss_LS, self).__init__()
        
        self.u_model = u_model
        self.u_model_LS = u_model_LS
        
        # The regularization parameters
        self.regul = regul
        
        # Number of neurons 
        self.nn = self.u_model.layers[-2].output_shape[1]
        
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
        
        self.laplac = tf.convert_to_tensor([[np.pi*((kx**2)/ lenx**2+(ky**2)/ leny**2)**0.5
                                          for kx in range(1,n_modesx+1)] for ky in range(1,n_modesy+1)],dtype=dtype)

        self.f = f
        self.n_modesx = n_modesx
        self.n_modesy = n_modesy
        
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.ptsx[self.select]
        y = self.ptsy[self.select]
        
        X,Y = tf.meshgrid(x,y)
        
        Xx = tf.reshape(X,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        Yy = tf.reshape(Y,[2*(self.n_ptsx-1)*2*(self.n_ptsy-1)])
        
        u = tf.unstack(self.u_model_LS(tf.stack([Xx,Yy],axis=-1)),axis=-1)
        u = tf.reshape(u,(self.nn,2*(self.n_ptsy-1),2*(self.n_ptsx-1)))
        
        ## Take appropriate transforms of each component
        FT_u = tf.einsum("kij,Jj,Ii,IJ->IJk", u,self.DST_x[self.select],
                         self.DST_y[self.select], self.laplac)
        mat_A = tf.reshape(FT_u,(self.n_modesx*self.n_modesy,self.nn))
        
        f_eval = tf.reshape(self.f(Xx,Yy),[2*(self.n_ptsy-1),2*(self.n_ptsx-1)])
        
        vecB = tf.einsum("ij,Jj,Ii,IJ->IJ",f_eval,self.DST_x[self.select],
                         self.DST_y[self.select],self.coeffs)
       
        aux = tf.reshape(vecB,(self.n_modesx*self.n_modesy,1))
        
        # The LS solution: The regularization is very important/sensitive here!!
        solution_w0 = tf.squeeze(tf.linalg.lstsq(mat_A,-aux,l2_regularizer=self.regul))
         
        # Assign the weights
        self.u_model.layers[-1].vars.assign(solution_w0)
        #hola = self.u_model.layers[-1].vars
        
        #FT_high = tf.einsum("IJk,k->IJ",FT_u,hola)
        FT_high = tf.einsum("ik,k->i",mat_A,solution_w0)
    
        # The value of the loss function
        return tf.reduce_sum((FT_high+tf.reshape(vecB,(1,self.n_modesx*self.n_modesy)))**2)
