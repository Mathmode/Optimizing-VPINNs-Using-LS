#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:57:12 2023

@author: Manuela Bastidas
"""
import tensorflow as tf
import numpy as np

# =============================================================================
# Plot the results
# =============================================================================

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


    
# =============================================================================
# PLOTS - SOLUTION - LOSS & ERROR
# =============================================================================

def plot_loss(history0,history1,exact_norm,final_directory):

    #Direct input 
    W = 5.4/2   # Figure width in inches, approximately A4-width - 2*1.25in margin
    plt.rcParams.update({
        'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
        'font.family': 'serif',
        #'font.sans-serif': 'Helvetica',
        'font.size' : 11,                   # Set font size to 11pt
        'axes.labelsize': 11,               # -> axis labels
        'xtick.labelsize' : 9,
        'ytick.labelsize' : 9,
        'legend.fontsize': 11,              # -> legends
        'mathtext.fontset' : 'cm',#computer moder
        'lines.linewidth': 0.5
    })
    
    ## ---------
    # Loss evolution
    ## ---------
    
    fig, ax = plt.subplots()
    
    # Plot the loss
   
    # plt.plot(history['sqrt_val'][1:]/exact_norm, color='b', linestyle= '--',
    #           label='Validation')
    
    diffloss = np.abs(tf.sqrt(history0['loss'][1:])/exact_norm-tf.sqrt(history1['loss'][1:])/exact_norm)
    
    plt.plot(diffloss, color='b', 
             #label='$\sqrt{\mathcal{L}(u)}$')
             label=r'losses')
             #label = 'Training')
    
    differror = np.abs((history0['error'][1:])/exact_norm-(history1['error'][1:])/exact_norm)
    
    plt.plot(differror, color='r', linestyle= ':',
              marker = 'o',markevery=0.05,markersize=1, 
              #label='$\|u-u^*\|_{H_0^1}$')
              label=r'errors')
              #label='$H_0^1$-error')
    
    plt.xlabel('Iterations')
    #plt.ylabel('$\sqrt{\mathcal{L}(u)}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.legend(frameon=False,handlelength=0,labelcolor='linecolor',loc='lower left')
    
    ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
    plt.tight_layout()
    
    #plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
    
    plt.savefig(f'{final_directory}/loss.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    
    ## ---------
    # Loss evolution
    ## ---------
    
    # fig, ax = plt.subplots()
    
    # # Plot the loss
    # plt.plot(history['error'][1:], color='k', linestyle= ':',
    #          marker = 'o',markevery=0.1, 
    #          #label='$\|u-u^*\|_{H_0^1}$')
    #          label=r'$\frac{\|u-u^*\|_{H_0^1}}{\|u^*\|_{H_0^1}}$')
    
    # # plt.plot(history['sqrt_val'][1:]/exact_norm, color='b', linestyle= '--',
    # #          label='Validation')
    
    # # plt.plot(tf.sqrt(history['loss'][1:])/exact_norm, color='r', 
    # #          #label='$\sqrt{\mathcal{L}(u)}$')
    # #          label=r'$\frac{\sqrt{\mathcal{L}(u)}}{\|u^*\|_{H_0^1}}$')
    
    # plt.xlabel('Iterations')
    # #plt.ylabel('$\sqrt{\mathcal{L}(u)}$')
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # plt.legend(loc ='lower left')
    
    # ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
    # plt.tight_layout()
    
    # #plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
    
    # plt.savefig(f'{final_directory}/error.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    # plt.show()
