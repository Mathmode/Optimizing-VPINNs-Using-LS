#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:56:57 2023

@author: Manuela Bastidas
"""

import tensorflow as tf
import numpy as np


from SCR.integration import integration_points_and_weights

# =============================================================================
# The DFR loss function
# weak implementation
# =============================================================================

class weak_loss(tf.keras.layers.Layer):
    def __init__(self,u_model,n_pts,n_modes,f,nrules,dtype='float64',**kwargs):
        super(weak_loss,self).__init__()
        
        self.u_model = u_model
        
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
            
        self.DCT = tf.constant(np.swapaxes([V*(k*np.pi/lenx)*np.cos(np.pi*k*(self.pts-a)/lenx)*W
                                             for k in range(1, n_modes+1)], 0, 1))
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([(lenx/(np.pi*k)) for k in range(1,n_modes+1)])

        self.f = f
       
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.pts[self.select]
        
        # Evaluate u and its derivative at integration points
        ## Persistent True not necessary because it only evaluates u'(once)
        with tf.GradientTape() as t1:
            t1.watch(x)
            u = self.u_model(x)
        du = t1.gradient(u,x)
        del t1

        ## The LHS of the loss function
        FT_high = tf.einsum("ji,i->j",self.DCT[self.select],du)
        
        # The RHS of the loss function
        FT_low = tf.einsum("ji,i->j",self.DST[self.select],self.f(x))

        ## Add and multiply by weighting factors
        FT_tot = (FT_high+FT_low)*self.coeffs
        
        # Loss calculation
        return tf.reduce_sum(FT_tot**2)
    
# =============================================================================
# The DFR loss function
# weak implementation
# =============================================================================

class ultraweak_loss(tf.keras.layers.Layer):
    def __init__(self,u_model,n_pts,n_modes,f,nrules,dtype='float64',**kwargs):
        super(ultraweak_loss,self).__init__()
    
        self.u_model = u_model
        
        # The domain is (0,pi) by default,include as input is the domain change
        b = np.pi
        a = 0
        lenx = b-a
        
        self.num_rules = nrules #negative number indicates a fix rule DST/DCT
        self.select = tf.Variable(0,dtype='int64')
        
        # Generate integration points
        pts, W = integration_points_and_weights(a,b,n_pts,self.num_rules,1)
        self.pts = tf.constant(pts)
            
        V = np.sqrt(2./lenx)
        self.DST = tf.constant(np.swapaxes([V*np.sin(np.pi*k*(self.pts-a)/lenx)*W
                                                for k in range(1,n_modes+1)], 0, 1))
            
        #self.DCT = tf.constant(np.swapaxes([V*(k*np.pi/lenx)*np.cos(np.pi*k*(self.pts-a)/lenx)*W
        #                                     for k in range(1, n_modes+1)], 0, 1))
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([(lenx/(np.pi*k)) for k in range(1,n_modes+1)])
        
        # Laplacian constant (eigenvalue)
        self.laplac =  np.array([((np.pi*k)/lenx)
                                for k in range(1, n_modes+1)]) 

        self.f = f
       
    def call(self,inputs):
        
        if self.num_rules>0:
            self.select.assign((self.select+1)%self.num_rules)
        
        x = self.pts[self.select]
        
        # Evaluating the model
        u = self.u_model(x)

        ## Take appropriate transforms of each component
        FT_high = tf.einsum("ji,i->j",self.DST[self.select],u)*self.laplac
        
        # The RHS of the loss function
        FT_low = tf.einsum("ji,i->j",self.DST[self.select],self.f(x))* self.coeffs
    
        ##Return the loss function 
        return tf.reduce_sum((FT_high+FT_low)**2)

    
## DFR loss function ( loss into layer )
#Import solution model
class error(tf.keras.layers.Layer):
    def __init__(self,u_model,n_pts,du_exact,dtype='float64',**kwargs):
        super(error,self).__init__()
        
        self.u_model = u_model
        
        # The domain is (0,pi) by default,include as input is the domain change
        b = np.pi
        #a = 0
        
        # Generate integration points
        pts, W = integration_points_and_weights(0,b,n_pts,1,1)
        self.eval_pts = tf.constant(pts[0])
        self.diff = W
        
        # Exact derivative
        self.du_exact = du_exact(self.eval_pts)

        ## H01 norm of exact solution (IN CASE RELATIVE - Change last line)
        # Ideally not calculated in smooth cases to see one to one relation
        # loss vs error
        self.H01norm = tf.reduce_sum(self.du_exact**2*self.diff)
       
    def call(self,inputs):
        
        ## Evaluate u and its derivative at integration points
        ## Persistent True not necessary because it only evaluates u'(once)
        with tf.GradientTape() as t1:
              t1.watch(self.eval_pts)
              u  = self.u_model(self.eval_pts)
        du = t1.gradient(u,self.eval_pts)
        del t1
        
        # The error in the norm H0_1
        H01err = tf.reduce_sum((du-self.du_exact)**2*self.diff)
        #L2ERROR = tf.reduce_sum((u-self.u_exact)**2*self.diff)
        
        #exact_norm = np.sqrt(11.3759)
        # aa = tf.abs(self.H01norm**0.5-np.sqrt(11.3759))/np.sqrt(11.3759)*100
    
        return H01err**0.5  #tf.abs(self.H01norm**0.5 -exact_norm)/exact_norm 
