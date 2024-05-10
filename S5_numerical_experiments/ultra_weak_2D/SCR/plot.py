#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:57:12 2023

@author: Manuela Bastidas
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Plot the results
# =============================================================================

from matplotlib import rcParams

#rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['legend.fontsize'] = 18
rcParams['mathtext.fontset'] = 'cm' 
rcParams['axes.labelsize'] = 20

# =============================================================================
# PLOTS - SOLUTION 
# =============================================================================

def plot_sol(u_model,exact_u,dux_exact,duy_exact,nn,WEAK,LS,final_directory):

    plot_npts = 1000

    hi_pts = np.linspace(0,np.pi,plot_npts+1)

    diff = tf.abs(hi_pts[1:] - hi_pts[:-1])
    eval_pts = tf.constant(hi_pts[:-1]+ diff/2, dtype='float64')

    x,y = tf.meshgrid(eval_pts,eval_pts)

    xt = tf.reshape(x,[plot_npts*plot_npts])
    yt = tf.reshape(y,[plot_npts*plot_npts])

    # Evaluate the derivative
    ue = exact_u(xt,yt)
    duex = dux_exact(xt,yt)
    duey = dux_exact(xt,yt)

    with tf.GradientTape(persistent=True) as t1:
        t1.watch(xt)
        t1.watch(yt)
        u = u_model(tf.stack([xt,yt],axis=-1))
        
    dux = t1.gradient(u,xt)
    duy = t1.gradient(u,yt)
    del t1
    
    Z_exact = tf.reshape(ue,[plot_npts,plot_npts])
    Z_app = tf.reshape(u,[plot_npts,plot_npts])
    Z_error = abs(Z_exact-Z_app)
    
    #M_duexact = np.sqrt(tf.reshape(duex,[plot_npts,plot_npts])**2+tf.reshape(duey,[plot_npts,plot_npts])**2)
    M_du = np.sqrt(tf.reshape(dux,[plot_npts,plot_npts])**2+tf.reshape(duy,[plot_npts,plot_npts])**2)
    M_error = np.sqrt((tf.reshape(dux,[plot_npts,plot_npts])-tf.reshape(duex,[plot_npts,plot_npts]))**2+
                      (tf.reshape(duy,[plot_npts,plot_npts])-tf.reshape(duey,[plot_npts,plot_npts]))**2)

    # fig, ax = plt.subplots()
    # # Exact solution
    # plt.contourf(x,y,Z_exact,cmap = 'coolwarm') #,levels=np.linspace(-1,1,5))
    
    # cbar = plt.colorbar(format = '%.1f')
    # cbar.ax.locator_params(nbins = 5)

    # plt.title('$u^*$',fontsize=12)
    
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')

    # plt.xticks([])
    # plt.yticks([])

    # ax.set_aspect('equal', 'box')

    # plt.tight_layout()
    # plt.savefig(f'{final_directory}/ref_solution.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    # plt.show()
    
    # ========================
    ## Solution 
    # ========================
    
    fig, ax = plt.subplots()
    # Exact solution
    plt.contourf(x,y,Z_app,cmap = 'coolwarm') #,levels=np.linspace(-1,1,5))

    cbar = plt.colorbar(format = '%.1f')
    cbar.ax.locator_params(nbins = 5)

    plt.title(f'Weak:{WEAK} & LS:{LS} & Nlast:{nn} \n $u$',fontsize=12)
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.xticks([])
    plt.yticks([])

    ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig(f'{final_directory}/solution.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    # ========================
    ## Error solution 
    # ========================
    
    fig, ax = plt.subplots()
   
    # Exact solution
    plt.contourf(x,y,Z_error,cmap = 'coolwarm')
    
    cbar = plt.colorbar(format = '%1.1e')
    cbar.ax.locator_params(nbins = 5)
   
    plt.title(f'Weak:{WEAK} & LS:{LS} & Nlast:{nn} \n $|u-u^*|$',fontsize=12)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    #plt.clim(0,tf.reduce_max(Z_exact))
    
    plt.xticks([])
    plt.yticks([])
    
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(f'{final_directory}/solution_error.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    # ===================
    # Solution DERIVATE
    # ===================
    
    fig, ax = plt.subplots()
    
    # Exact solution
    plt.contourf(x,y,M_du,cmap = 'coolwarm')
    
    cbar = plt.colorbar(format = '%.1f')
    cbar.ax.locator_params(nbins = 5)
   
    plt.title(f'Weak:{WEAK} & LS:{LS} & Nlast:{nn} \n $\|grad(u)\|$ ',fontsize=12)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
    plt.xticks([])
    plt.yticks([])
    
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(f'{final_directory}/grad_solution.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()
    
    # ===================
    # Error Grad solution 
    # ===================
    
    fig, ax = plt.subplots()
    
    # Exact solution
    plt.contourf(x,y,M_error,cmap = 'coolwarm')
    
    cbar = plt.colorbar(format = '%1.1e')
    cbar.ax.locator_params(nbins = 5)
   
    plt.title(f'Weak:{WEAK} & LS:{LS} & Nlast:{nn} \n $\|grad(u-u^*)\|$',fontsize=12)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
    plt.xticks([])
    plt.yticks([])
    
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(f'{final_directory}/grad_error.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()


def plot_relError(u_model,ref_norm,history,nn_last,WEAK,LS,final_directory):
    
    fig, ax = plt.subplots()
    
    # Plot the loss
    plt.plot(history.history['error']/tf.constant(ref_norm)*100, color='b', linestyle= '-',
             marker = 'o',markevery=0.1)
   
    plt.xlabel('Iterations')
    plt.ylabel('Relative $H_0^1$-error (%)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    #plt.legend(loc ='lower left')
    
    ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
    plt.tight_layout()
    
    plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn_last}',fontsize=12)
    
    plt.savefig(f'{final_directory}/rel_error.pdf', format = 'pdf', bbox_inches = 0, transparent = True)
    plt.show()


    
# =============================================================================
# PLOTS - SOLUTION - LOSS & ERROR
# =============================================================================

# def plot_loss(u_model,exact_u,history,nn,WEAK,LS,final_directory):

    
#     ## ---------
#     # Loss evolution
#     ## ---------
    
#     fig, ax = plt.subplots()
    
#     # Plot the loss
#     plt.plot(history.history['error'], color='gray', linestyle= '--',
#              marker = 'o',markevery=0.1, label='$\|u-u^*\|_{H_0^1}$')
#     plt.plot(history.history['sqrt_val'], color='b', linestyle= '--',
#              label='Validation')
    
#     plt.plot(tf.sqrt(history.history['loss']), color='r', label='Training')
    
#     plt.xlabel('Iterations')
#     plt.ylabel('$\sqrt{\mathcal{L}(u)}$')
    
#     ax.set_xscale('log')
#     ax.set_yscale('log')
    
#     plt.legend(loc ='lower left')
    
#     ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
#     #plt.tight_layout()
    
#     plt.title(f'Weak:{WEAK} & LS:{LS} & $N_l$:{nn}',fontsize=12)
    
#     plt.savefig(f'{final_directory}/W{WEAK}_LS{LS}_Nl{nn}.pdf', format = 'pdf', bbox_inches = 'tight', transparent = True)
#     plt.show()


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
