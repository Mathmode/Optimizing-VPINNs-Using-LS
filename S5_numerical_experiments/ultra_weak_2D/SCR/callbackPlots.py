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

from matplotlib.cm import ScalarMappable

class PlotSolutionCallback(tf.keras.callbacks.Callback):
    def __init__(self,u_true,dux_true,duy_true,save_figs):
        super(PlotSolutionCallback, self).__init__()
        
        plot_npts = 1000
        x_data = tf.constant(np.linspace(0,np.pi,plot_npts))
        y_data = tf.constant(np.linspace(0,np.pi,plot_npts))
        
        self.x,self.y = tf.meshgrid(x_data,y_data)

        xt = tf.reshape(self.x,[plot_npts*plot_npts])
        yt = tf.reshape(self.y,[plot_npts*plot_npts])

        # Evaluate the derivative
        self.ue = u_true(xt,yt)
        self.duex = dux_true(xt,yt)
        self.duey = duy_true(xt,yt)
    
        self.save_figs = save_figs
        self.save_num = tf.Variable(0,dtype='int32')

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == 1 or epoch%20 == 0:
            
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
            
            plot_npts = 1000
            
            xt = tf.reshape(self.x,[plot_npts*plot_npts])
            yt = tf.reshape(self.y,[plot_npts*plot_npts])
            
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(xt)
                t1.watch(yt)
                
                u = self.model.layers[1].u_model(tf.stack([xt,yt],axis=-1))
                
            dux = t1.gradient(u,xt)
            duy = t1.gradient(u,yt)
            del t1
            
            Z_exact = tf.reshape(self.ue,[plot_npts,plot_npts])
            Z_app = tf.reshape(u,[plot_npts,plot_npts])
            Z_error = abs(Z_exact-Z_app)
            
            #M_duexact = np.sqrt(tf.reshape(duex,[plot_npts,plot_npts])**2+tf.reshape(duey,[plot_npts,plot_npts])**2)
            M_du = np.sqrt(tf.reshape(dux,[plot_npts,plot_npts])**2 + 
                           tf.reshape(duy,[plot_npts,plot_npts])**2)
            
            M_error = np.sqrt((tf.reshape(dux,[plot_npts,plot_npts]) - 
                               tf.reshape(self.duex,[plot_npts,plot_npts]))**2 +
                              (tf.reshape(duy,[plot_npts,plot_npts]) - 
                               tf.reshape(self.duey,[plot_npts,plot_npts]))**2)
            
            ## ---------
            # SOLUTION
            ## ---------
            
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.contourf(self.x,self.y,Z_app,cmap = 'coolwarm',
                         levels=np.linspace(-1.2,1.2,25), extend='both')

            cbar = plt.colorbar(format = '%.1f')
            cbar.ax.locator_params(nbins = 5)
            
            plt.xticks([])
            plt.yticks([])
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            plt.title(f'Solution $u$ \n Iteration {epoch}')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_figs}/solution-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
            ## ---------
            # Error SOLUTION
            ## ---------
            
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.contourf(self.x,self.y,Z_error,cmap = 'coolwarm',
                         levels=np.linspace(0,0.5,25), extend='max')

            cbar = plt.colorbar(format = '%.2f')
            cbar.ax.locator_params(nbins = 5)
            
            plt.xticks([])
            plt.yticks([])
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            plt.title(f'$|u^*-u|$ \n Iteration {epoch}')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_figs}/error_solution-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
            ## ---------
            # GRAD SOLUTION
            ## ---------
            
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.contourf(self.x,self.y,M_du,cmap = 'coolwarm',
                         levels=np.linspace(0,4,10), extend='both')

            cbar = plt.colorbar(format = '%.1f')
            cbar.ax.locator_params(nbins = 5)
            
            plt.xticks([])
            plt.yticks([])
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            plt.title(f'Gradient solution $\\nabla u$ \n Iteration {epoch}')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_figs}/gradsol-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
            ## ---------
            # Error GRAD SOLUTION
            ## ---------
            
            fig, ax = plt.subplots()
            
            # Plot the approximate solution obtained from the trained model
            plt.contourf(self.x,self.y,M_error,cmap = 'coolwarm',
                         levels=np.linspace(0,0.5,25), extend='max')

            cbar = plt.colorbar(format = '%.2f')
            cbar.ax.locator_params(nbins = 5)
            
            plt.xticks([])
            plt.yticks([])
            
            ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
            
            plt.title(f'$\|\\nabla (u^*-u)\|$ \n Iteration {epoch}')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_figs}/error_gradsol-{self.save_num.numpy()}.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
            #plt.show()
            
            self.save_num.assign_add(1)
            
    