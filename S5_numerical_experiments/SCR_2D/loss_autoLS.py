# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:35:25 2023

@author: jamie.taylor
"""

import tensorflow as tf

# =============================================================================
# # Makes the LS loss model
# This function wraps the model of the loss to compute the LS solver of 
    # the last layer
# =============================================================================

def model_autoLS(loss_model,regul,dtype='float64'):
    
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    # The automatic loss layer implementation
    loss_l = ls_loss_layer(loss_model,regul,dtype=dtype)(xvals)
   
    min_model = tf.keras.Model(inputs=xvals,outputs=loss_l)
    
    min_model.summary()
    
    return min_model 

# =============================================================================
# Loss layer to compute the automatic LS
# =============================================================================
# This layer computes the LS solver of the last layer of the loss_model

class ls_loss_layer(tf.keras.layers.Layer):    
    def __init__(self,loss_model,regul,dtype='float64',**kwargs):
        super(ls_loss_layer, self).__init__()
        
        self.loss_model = loss_model
        self.u_model = self.loss_model.layers[1].u_model
        
        self.nn = self.u_model.layers[-2].output_shape[1]
        
        #self.regul = regul
        self.regul = tf.eye(self.nn,self.nn,dtype=dtype)*regul
        
    def call(self,inputs):
        
        # Assign the weights to be 0
        self.u_model.layers[-1].vars.assign(tf.zeros([self.nn],dtype='float64'))
        
        # Weights of the last layer
        w_last = self.u_model.layers[-1].vars
        
        # Gradient of the loss function wrt the weights of the last layer 
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(w_last)
           
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(w_last)
                
                # The output of the loss model has 4 outputs if VAL and ERR
                outputloss = self.loss_model(tf.constant([1.]))
                loss_val = outputloss[0]
                
            # This method implies computing two derivatives
            # B = the RHS of the system -- > Atb  
            B = t2.gradient(loss_val,w_last)
        # B = the LHS of the system -- > AtA 
        A = t1.jacobian(B,w_last)
        
        w_new = tf.squeeze(tf.linalg.solve(A+2*self.regul,-tf.reshape(B,[self.nn,1])))
        
        # w_new = tf.reshape(tf.linalg.lstsq(ddl,-tf.reshape(b,[nn+1,1]),l2_regularizer=lam),[nn+1])
        self.u_model.layers[-1].vars.assign(w_new)
        
        # Evaluation of the loss function (Once more!) 
            # This happens because the system solved here corresponds to the normal eq
        return self.loss_model(tf.constant([1.])) 
        


    
    
    