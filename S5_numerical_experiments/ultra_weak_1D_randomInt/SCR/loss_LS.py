#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author:  Manuela Bastidas
"""

import tensorflow as tf
import numpy as np

from SCR.integration import integration_points_and_weights

# =============================================================================
# The DFR loss function
# weak implementation and solving the LS system
# =============================================================================

# Here we import the u model
class weak_loss_LS(tf.keras.layers.Layer):
    def __init__(self,u_model,u_model_LS,regul,n_pts,n_modes,f,nrules,dtype='float64',**kwargs):
        super(weak_loss_LS,self).__init__()
        
        # saving the model and the model until the last layer
        self.u_model = u_model
        self.u_model_LS = u_model_LS
        
        # The regularization parameters
        self.regul = regul
        
        # Number of neurons 
        self.nn = self.u_model.layers[-2].output_shape[1]
        
        # The domain is (0,pi) by default,include as input if the domain change
        b = np.pi
        a = 0
        lenx = b-a
        
        self.num_rules = nrules #negative number indicates a fix rule DST/DCT
        self.select = tf.Variable(0,dtype='int32')
        
        # Generate integration points
        pts, W = integration_points_and_weights(a,b,n_pts,self.num_rules,1)
        self.pts = tf.constant(pts)
        
        V = np.sqrt(2./lenx)
        self.DST = tf.constant(np.swapaxes([V*np.sin(np.pi*k*(self.pts-a)/lenx)*W
                                                for k in range(1,n_modes+1)], 0, 1))
            
        self.DCT = tf.constant(np.swapaxes([V*(k*np.pi/lenx)*np.cos(np.pi*k*(self.pts-a)/lenx)*W
                                             for k in range(1, n_modes+1)], 0, 1))
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([(lenx/(np.pi*k)) for k in range(1,n_modes+1)])

        self.f = f

        # Number of modes
        self.n_modes = n_modes
    
        
    def call(self,inputs):
        
        # # Evaluate u and its derivative at integration points
        # with tf.GradientTape(persistent=True) as t1:
        #     t1.watch(self.pts)
        #     # Last layer evaluation
        #     u = tf.unstack(self.u_model_LS(self.pts),axis=-1)
        
        # # Derivative of the last layer functions
        # # Improvements: (i) Jacobian (not the usual, it is vectorial function and scalar evaluation)
        # # (ii) batch_jacobian: Imply reshaping pts and (iii) use matv instead of for 
        # du = tf.stack([t1.gradient(u[i],self.pts) for i in range(self.nn)])
        # del t1 
        #select = tf.random.uniform([1],dtype='int64',maxval=self.num_rules)[0]
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
         
        x = self.pts[self.select]
        
        with tf.autodiff.ForwardAccumulator(
                primals = x,
                tangents = tf.ones_like(x)) as t1:
            u = self.u_model_LS(x)
        du = t1.jvp(u)
        del t1
        
        # The "traditional" LHS of the weak/ultraweak formulation (Nneurons x Nmodes)
        mat_A = tf.einsum("ji,ik,j->jk",self.DCT[self.select],du,self.coeffs)
        
        # The RHS vector of the weak/ultraweak formulations
        vec_B = tf.einsum("ji,i,j->j",self.DST[self.select],self.f(x),self.coeffs)

        # The LS solution: The regularization is very important/sensitive here!!
        solution_w0 = tf.squeeze(tf.linalg.lstsq(mat_A,
                                                 -tf.reshape(vec_B,(self.n_modes,1)), 
                                                 l2_regularizer=self.regul))
         
        # Assign the weights
        self.u_model.layers[-1].vars.assign(solution_w0)
        
        #w_last = self.u_model.layers[-1].vars
        # Ax : The RHS of the weak/ultraweak formulation
        FT_high = tf.einsum("ji,i->j",mat_A,solution_w0)
        
        # The value of the loss function
        return tf.reduce_sum((FT_high+vec_B)**2)
    

# =============================================================================
# The DFR loss function
# ultraweak implementation and solving the LS system
# =============================================================================

class ultraweak_loss_LS(tf.keras.layers.Layer):
    def __init__(self,u_model,u_model_LS,regul,n_pts,n_modes,f,nrules,dtype='float64',**kwargs):
        super(ultraweak_loss_LS,self).__init__()
        
        # saving the model and the model until the last layer
        self.u_model = u_model
        self.u_model_LS = u_model_LS
        
        # regularization parameter
        self.regul = regul
        
        # Number of neurons in the last layer
        #self.nn = self.u_model.layers[-2].output_shape[1]
        
        # The domain is (0,pi) by default,include as input is the domain change
        b = np.pi
        a = 0
        lenx = b-a
        
        self.num_rules = nrules #negative number indicates a fix rule DST/DCT
        self.select = tf.Variable(0,dtype='int32')
        
        # Generate integration points
        pts, W = integration_points_and_weights(a,b,n_pts,self.num_rules,1)
        self.pts = tf.constant(pts)
            
        V = np.sqrt(2./lenx)
        self.DST = tf.constant(np.swapaxes([V*np.sin(np.pi*k*(self.pts-a)/lenx)*W
                                                for k in range(1,n_modes+1)], 0, 1))
            
       # self.DCT = tf.constant(np.swapaxes([V*(k*np.pi/lenx)*np.cos(np.pi*k*(self.pts-a)/lenx)*W
                                             #for k in range(1, n_modes+1)], 0, 1))
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([(lenx/(np.pi*k)) for k in range(1,n_modes+1)])

        self.f = f

        # The eigenvalues - constant multiplying the laplacian 
        self.laplac = tf.constant([((np.pi*k)/lenx)
                                for k in range(1,n_modes+1)],dtype=dtype)
       
        self.n_modes = n_modes
       
    def call(self,inputs):
        
        # Last layer evaluation (this includes the BC)
        #select = tf.random.uniform([1],dtype='int64',maxval=self.num_rules)[0]
    
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.pts[self.select]
        
        u = self.u_model_LS(x)
        
        # The matrix of the LHS (Nneurons x nmodes)
        mat_A = tf.einsum("ji,ik,j->jk",self.DST[self.select],u,self.laplac)
        
        # The RHS
        vec_B = tf.einsum("ji,i,j->j",self.DST[self.select],self.f(x),self.coeffs)
      
        # Solution of the LS 
        solution_w0 = tf.squeeze(tf.linalg.lstsq(mat_A,
                                                 -tf.reshape(vec_B,(self.n_modes,1)),
                                                 l2_regularizer=self.regul))
        
        # Assign the weights
        self.u_model.layers[-1].vars.assign(solution_w0)
        
        # Ax
        FT_high = tf.einsum("ji,i->j",mat_A,solution_w0)
        
        # The value of the loss function
        return tf.reduce_sum((FT_high+vec_B)**2)
    
