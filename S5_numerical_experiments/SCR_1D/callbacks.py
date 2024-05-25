#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) 


class EndOfTrainingLS(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, logs=None):
        
        inputs = self.model.train_data()
        
        self.model.construct_LHMandRHV(inputs)
        self.model.optimal_computable_vars()