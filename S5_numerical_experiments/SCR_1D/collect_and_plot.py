#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Customize other style parameters
plt.rcParams.update({
    'font.size': 12,               # Set the font size for text
    'axes.labelsize': 12,          # Set the font size for axis labels
    'legend.fontsize': 12,         # Set the font size for legends
    'xtick.labelsize': 10,         # Set the font size for x-axis tick labels
    'ytick.labelsize': 10,         # Set the font size for y-axis tick labels
    'axes.linewidth': 0.8,         # Set the line width for axes
    'lines.linewidth': 1.5,        # Set the line width for plot lines
    'legend.frameon': True,       # Disable legend frame
    'legend.loc': 'best',          # Set legend location
    'text.latex.preamble': r'\usepackage{amsmath, amssymb}'  # For using AMS LaTeX package if needed
})

from config import M,EXACT
from SCR_1D.models import error, residual

def auto_format_y_axis():
    
    # get the current axis
    ax = plt.gca()
    
    # Get the current y-axis limits
    y_min, y_max = ax.get_ylim()
    
    # Check if y-axis limits are in the form of small decimal values
    if abs(y_min) < 1 and abs(y_max) < 1 and y_min != y_max:
        exponent = int(np.floor(np.log10(max(abs(y_min), abs(y_max)))))
        
        # Only apply the formatting if the exponent is in the range we care about
        if exponent < 0:
            scale_factor = 10 ** -exponent
            
            # Set the formatter for the y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val * scale_factor:.2f}'))
            
            # Add the scale label above the y-axis
            ax.text(0.01, 1.01, r'$\times 10^{{{}}}$'.format(exponent), transform=ax.transAxes, ha='left', va='bottom')

    

def save_and_plot_net(x, nets, file=False):
    
    figures = []
    
    netGD, netLSGD = nets
    
    exact_out = EXACT(x)
    
    data = {'x': x.numpy().flatten(),
            'exact': exact_out.numpy().flatten()
            }
    
    if netGD != None:
        errorGD = error(netGD)
        residualGD = residual(netGD)
        netGD_out = netGD(x)
        errorGD_out = errorGD(x)
        derrorGD_out = errorGD.d(x)
        residualGD_out = residualGD(x)
        dresidualGD_out = residualGD.d(x)
        
        data['netGD'] = netGD_out.numpy().flatten()
        data['errorGD'] = errorGD_out.numpy().flatten()
        data['derrorGD'] = derrorGD_out.numpy().flatten()
        data['residualGD'] = residualGD_out.numpy().flatten()
        data['dresidualGD'] = dresidualGD_out.numpy().flatten()
        data['resminuserrGD'] = residualGD_out.numpy().flatten() - errorGD_out.numpy().flatten()
        data['dresminuserrGD'] = dresidualGD_out.numpy().flatten() - derrorGD_out.numpy().flatten()
        
        
    if netLSGD != None:
        errorLSGD = error(netLSGD)
        residualLSGD = residual(netLSGD)
        netLSGD_out = netLSGD(x)
        errorLSGD_out = errorLSGD(x)
        derrorLSGD_out = errorLSGD.d(x)
        residualLSGD_out = residualLSGD(x)
        dresidualLSGD_out = residualLSGD.d(x)
        
        data['netLSGD'] = netLSGD_out.numpy().flatten()
        data['errorLSGD']: errorLSGD_out.numpy().flatten()
        data['derrorLSGD']: derrorLSGD_out.numpy().flatten()
        data['residualLSGD'] = residualLSGD_out.numpy().flatten()
        data['dresidualGD'] = dresidualLSGD_out.numpy().flatten()
        data['resminuserrLSGD'] = residualLSGD_out.numpy().flatten() - errorLSGD_out.numpy().flatten()
        data['dresminuserrLSGD'] = dresidualLSGD_out.numpy().flatten() - derrorLSGD_out.numpy().flatten()
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    # plt.rcParams.update({
    #     'figure.figsize': (2.7,4.05),     # 4:3 aspect ratio
    #     'font.family': 'serif',
    #     #'font.sans-serif': 'Helvetica',
    #     'font.size' : 8,                   # Set font size to 11pt
    #     'axes.labelsize': 11,               # -> axis labels
    #     'xtick.labelsize' : 9,
    #     'ytick.labelsize' : 9,
    #     'legend.fontsize': 9,              # -> legends
    #     'mathtext.fontset' : 'cm',#computer moder
    #     'lines.linewidth': 0.5
    # })
    
    plot_width, plot_height = 6.4, 4.8
    # extra_space_above = 0.85
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,exact_out,label=r"$u^*$",color="C2",lw=4,alpha=0.5)
    plt.plot(x,netGD_out,label=r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}$ Adam", color="C1",lw=2)
    plt.plot(x,netLSGD_out,label=r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}$ LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}(x)$ vs. $u^*(x)$", labelpad=10)
    plt.title("Prediction vs. exact solution")
    figures.append(plt.gcf())
    plt.show()

    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,errorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,errorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}}) = u^{\boldsymbol{\alpha},\boldsymbol{\omega}}(x) - u^*(x)$", labelpad=10)
    plt.title("Error")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,residualGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,residualLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
    plt.title("Discretized residual")
    auto_format_y_axis()
    #plt.subplots_adjust(top=extra_space_above)
    figures.append(plt.gcf())
    plt.show()
    
        
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,residualGD_out - errorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,residualLSGD_out - errorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}}) - r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
    plt.title("Discretization error")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    # plt.figure(figsize=(plot_width, plot_height))
    # plt.plot(x,derrorGD_out,label="Adam",color="C1",lw=2)
    # plt.plot(x,derrorLSGD_out,label="LS/Adam",color="C0",lw=1)
    # plt.legend()
    # plt.xlabel(r"$x$")
    # plt.title(r"$\nabla e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    # auto_format_y_axis()
    # figures.append(plt.gcf())
    # plt.show()
    
    # plt.figure(figsize=(plot_width, plot_height))
    # plt.plot(x,dresidualGD_out,label="Adam",color="C1",lw=2)
    # plt.plot(x,dresidualLSGD_out,label="LS/Adam",color="C0",lw=1)
    # plt.legend()
    # plt.xlabel(r"$x$")
    # plt.title(r"$\nabla r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    # auto_format_y_axis()
    # figures.append(plt.gcf())
    # plt.show()
    
    # plt.figure(figsize=(plot_width, plot_height))
    # plt.plot(x,dresidualGD_out - derrorGD_out,label="Adam",color="C1",lw=2)
    # plt.plot(x,dresidualLSGD_out - derrorLSGD_out,label="LS/Adam",color="C0",lw=1)
    # plt.legend()
    # plt.xlabel(r"$x$")
    # plt.title(r"$\nabla\{r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}}) - r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\}$")
    # auto_format_y_axis()
    # figures.append(plt.gcf())
    # plt.show()
    
    return figures

def save_and_plot_spectrum(nets, file=False):
    
    figures = []
    
    netGD, netLSGD = nets
    
    data = dict()
    
    if netGD != None:
        residualGD = residual(netGD)
        modesGD, spectrumGD_out = residualGD.spectrum()
        
        data['modes'] = modesGD
        data['coeffGD'] = spectrumGD_out
        
        
    if netGD != None:
        residualLSGD = residual(netLSGD)
        modesLSGD, spectrumLSGD_out = residualLSGD.spectrum()
        
        data['modes'] = modesLSGD
        data['coeffLSGD'] = spectrumLSGD_out
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    plot_width, plot_height = 8.4, 4.8
    
            
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(modesGD,spectrumGD_out,'o', markersize=3, label="Adam",color="C1")
    plt.plot(modesLSGD,spectrumLSGD_out,'o', markersize=2, label="LS/Adam",color="C0")
    plt.ylabel(r"$r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_m)$", labelpad=10)
    plt.xlabel(r"$m$")
    plt.yscale("symlog", linthresh=10**(-8))
    plt.axvline(x=M, color='r', linestyle='--', label=rf"cut-off $M={M}$")
    plt.legend()
    plt.title("Spectrum of the residual")
    figures.append(plt.gcf())
    plt.show()
    
    return figures
    
def save_and_plot_loss(history, file=False):
    
    figures = []
    
    data = dict()
    iterations = np.arange(1,len(history["lossGD"])+1)
    data['iteration'] = iterations
    data['lossGD'] = history["lossGD"]
    data['lossLSGD'] = history["lossLSGD"]
    data['errorGD'] = history["errorGD"]
    data['errorLSGD'] = history["errorLSGD"]
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    plot_width, plot_height = 8.4, 4.8
            
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(iterations, history["lossGD"],label=r"$\mathcal{L}(\boldsymbol{\alpha},\boldsymbol{\omega}) = \Vert r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{V}}^2$ Adam", color="C1",lw=3, alpha=0.5)
    plt.plot(iterations, history["lossLSGD"],label=r"$\mathcal{L}(\boldsymbol{\alpha},\boldsymbol{\omega}) = \Vert r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{V}}^2$ LS/Adam", color="C0",lw=3, alpha=0.5)
    plt.plot(iterations, history["errorGD"],label=r"$\Vert e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{U}}^2$ Adam", color="red", linestyle="--", lw=1)
    plt.plot(iterations, history["errorLSGD"],label=r"$\Vert e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{U}}^2$ LS/Adam", color="purple", linestyle="--", lw=1)
    plt.yscale("log")
    plt.ylabel("Loss and error in the energy norm", labelpad=10)
    plt.xlabel("Iteration")
    plt.legend()
    plt.title("Training history")
    figures.append(plt.gcf())
    plt.show()
    
    return figures