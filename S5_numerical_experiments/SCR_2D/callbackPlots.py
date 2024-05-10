#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:41:25 2023

@author: manuela
"""

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

import numpy as np
import tensorflow as tf

class PlotSolutionCallback(tf.keras.callbacks.Callback):
    def __init__(self,u_true,du_true,save_figs):
        super(PlotSolutionCallback, self).__init__()
        self.x_data = tf.constant(np.linspace(0,np.pi,1000))
        self.u_true = u_true(self.x_data)
        self.du_true = du_true(self.x_data)
        self.save_figs = save_figs
        self.save_num = tf.Variable(0,dtype='int32')

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == 1 or epoch%10 == 0:
            
            #Direct input 
            W = 5.4/2   # Figure width in inches, approximately A4-width - 2*1.25in margin
            plt.rcParams.update({
                'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
                'font.family': 'serif',
                #'font.sans-serif': 'Helvetica',
                'font.size' : 8,                   # Set font size to 11pt
                'axes.labelsize': 11,               # -> axis labels
                'xtick.labelsize' : 9,
                'ytick.labelsize' : 9,
                'legend.fontsize': 9,              # -> legends
                'mathtext.fontset' : 'cm',#computer moder
                'lines.linewidth': 0.5
            })
            
            # Plot the solution at each epoch
            x_train = self.model.layers[1].pts[self.model.layers[1].select]
            
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(self.x_data)
                t1.watch(x_train)
                u_all = self.model.layers[1].u_model(self.x_data)
                u = self.model.layers[1].u_model(x_train)
            du_all = t1.gradient(u_all, self.x_data)
            du = t1.gradient(u,x_train)
            del t1
            
            ## ---------
            # SOLUTION
            ## ---------
            
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.plot(self.x_data, self.u_true, color='r')
            
            plt.plot(self.x_data, u_all, color='b')
            plt.plot(x_train,u,color='k',linestyle='',markersize=1.5,marker='*',markevery=100,
                     label='_nolegend_')
            
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            
            #plt.ylim(-0.8, 0.4)
            
            plt.legend(['Approx. $u$', 'Exact $u^*$'],loc='lower left',frameon=True,handlelength=0,labelcolor='linecolor')
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            plt.title(f'Iteration {epoch}')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_figs}/solution-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
    
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.plot(self.x_data, self.du_true, color='r')
            
            plt.plot(self.x_data, du_all, color='b')
            plt.plot(x_train,du,color='k',linestyle='',markersize=1.5,marker='*',markevery=100,
                      label='_nolegend_')
            
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            
            #plt.ylim(-1.2, 1.8)
            
            plt.legend(['Approx. d$_x$$u$', 'Exact d$_x$$u^*$'],loc='upper center',frameon=True,handlelength=0,labelcolor='linecolor')
            #plt.legend(['$u$', '$u^*$'],frameon=False,handlelength=0,labelcolor='linecolor')
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            #plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            plt.title(f'Iteration {epoch}')
            
            plt.tight_layout()
            
            plt.savefig(f'{self.save_figs}/deriv-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
            self.save_num.assign_add(1)
            
    