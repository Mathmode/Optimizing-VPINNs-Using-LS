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
# PLOTS - SOLUTION 
# =============================================================================

def plot_sol(u_model,exact_u,du_exact,nn_last,WEAK,LS,final_directory):

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
    
    # Generate a list of x values for visualization
    xlist = tf.constant([np.pi/1000 * i for i in range(1000)],dtype='float64')
    
    ## ---------
    # SOLUTION
    ## ---------
    
    fig, ax = plt.subplots()
    # Plot the approximate solution obtained from the trained model
    plt.plot(xlist, u_model(xlist), color='b')
    plt.plot(xlist, u_model(xlist), linestyle='',color='k', marker='*', markersize=2, markevery=100,
             label='_nolegend_')
    plt.plot(xlist, exact_u(xlist), color='r')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    #plt.ylim(-0.8, 0.4)
    
    plt.legend(['Approx. $u$', 'Exact $u^*$'],loc='lower left',frameon=True,handlelength=0,labelcolor='linecolor')
   
    ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
   
    plt.title('Iteration 1000')#f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
   
    plt.tight_layout()
    
    plt.savefig(f'{final_directory}/solution-101.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    ## ---------
    # Derivative SOLUTION
    ## ---------
    
    with tf.GradientTape() as t1:
        t1.watch(xlist)
        uu = u_model(xlist)
    du = t1.gradient(uu, xlist)
    del t1
    
    fig, ax = plt.subplots()
    # Plot the approximate solution obtained from the trained model
    plt.plot(xlist, du, color='b')
    plt.plot(xlist, du,linestyle='',color='k', marker='*', markersize=2, markevery=100,
             label='_nolegend_')
    plt.plot(xlist, du_exact(xlist), color='r')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    #plt.ylim(-1.2, 1.8)
    
    plt.legend(['Approx. d$_x$$u$', 'Exact d$_x$$u^*$'],loc='upper center',frameon=True,handlelength=0,labelcolor='linecolor')
    #plt.legend(['$u$', '$u^*$'],frameon=False,handlelength=0,labelcolor='linecolor')
    
    ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
    
    #plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
    plt.title('Iteration 1000')
    
    plt.tight_layout()
    
    plt.savefig(f'{final_directory}/deriv-101.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    
# =============================================================================
# PLOTS - SOLUTION - LOSS & ERROR
# =============================================================================

def plot_loss(u_model,history,nn_last,WEAK,LS,exact_norm,final_directory):

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
   
    # plt.plot(history.history['sqrt_val'][1:]/exact_norm, color='b', linestyle= '--',
    #           label='Validation')
    
    plt.plot(tf.sqrt(history.history['loss'][1:])/exact_norm, color='r', 
             #label='$\sqrt{\mathcal{L}(u)}$')
             label=r'$\frac{\sqrt{\mathcal{L}(u)}}{\|u^*\|_{H_0^1}}$')
             #label = 'Training')
    
    plt.plot(history.history['error'][1:]/exact_norm, color='k', linestyle= ':',
              marker = 'o',markevery=0.05,markersize=1, 
              #label='$\|u-u^*\|_{H_0^1}$')
              label=r'$\frac{\|u-u^*\|_{H_0^1}}{\|u^*\|_{H_0^1}}$')
              #label='$H_0^1$-error')
    
    plt.xlabel('Iterations')
    #plt.ylabel('$\sqrt{\mathcal{L}(u)}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.legend(frameon=False,handlelength=0,labelcolor='linecolor')
    
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
    # plt.plot(history.history['error'][1:], color='k', linestyle= ':',
    #          marker = 'o',markevery=0.1, 
    #          #label='$\|u-u^*\|_{H_0^1}$')
    #          label=r'$\frac{\|u-u^*\|_{H_0^1}}{\|u^*\|_{H_0^1}}$')
    
    # # plt.plot(history.history['sqrt_val'][1:]/exact_norm, color='b', linestyle= '--',
    # #          label='Validation')
    
    # # plt.plot(tf.sqrt(history.history['loss'][1:])/exact_norm, color='r', 
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
