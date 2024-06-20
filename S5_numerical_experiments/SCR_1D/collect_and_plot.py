#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import LEGEND_LOCATION

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Customize other style parameters
plt.rcParams.update({
    'font.size': 15,               # Set the font size for text
    'axes.labelsize': 15,          # Set the font size for axis labels
    'legend.fontsize': 12,         # Set the font size for legends
    'xtick.labelsize': 15,         # Set the font size for x-axis tick labels
    'ytick.labelsize': 15,         # Set the font size for y-axis tick labels
    'axes.linewidth': 0.8,         # Set the line width for axes
    'lines.linewidth': 1.5,        # Set the line width for plot lines
    'legend.frameon': True,        # Enable legend frame
    'legend.loc': LEGEND_LOCATION, # Set legend location
    'text.latex.preamble': r'\usepackage{amsmath, amssymb}'  # LaTeX packages needed
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
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val * scale_factor:.1f}'))
            
            # Add the scale label above the y-axis
            ax.text(0.01, 1.01, r'$\times 10^{{{}}}$'.format(exponent), transform=ax.transAxes, ha='left', va='bottom')
                
# Define the formatter function
def thousands_formatter(x, pos):
    return f'{x:,.0f}'


def save_and_plot_net(x, nets, file=False):
    
    figures = []
    
    netINI, netGD, netLSGD = nets
    
    exact_out = EXACT(x)
    
    data = {'x': x.numpy().flatten(),
            'exact': exact_out.numpy().flatten()
            }
    
    if netINI != None:
        errorINI = error(netINI)
        residualINI = residual(netINI)
        netINI_out = netINI(x)
        errorINI_out = errorINI(x)
        derrorINI_out = errorINI.d(x)
        residualINI_out = residualINI(x)
        dresidualINI_out = residualINI.d(x)
        
        data['netINI'] = netINI_out.numpy().flatten()
        data['errorINI'] = errorINI_out.numpy().flatten()
        data['derrorINI'] = derrorINI_out.numpy().flatten()
        data['residualINI'] = residualINI_out.numpy().flatten()
        data['dresidualINI'] = dresidualINI_out.numpy().flatten()
        data['resminuserrINI'] = residualINI_out.numpy().flatten() - errorINI_out.numpy().flatten()
        data['dresminuserrINI'] = dresidualINI_out.numpy().flatten() - derrorINI_out.numpy().flatten()
        
    
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
    plt.ylabel(r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}(x) \quad \text{and} \quad u^*(x)$", labelpad=10)
    plt.title("Prediction vs. exact solution")
    figures.append(plt.gcf())
    plt.show()

    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,errorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,errorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
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
    plt.title("Discrete residual")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
        
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,residualGD_out - errorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,residualLSGD_out - errorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}}) - r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
    plt.title("Residual discretization error")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,derrorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,derrorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\nabla e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
    plt.title(r"Derivative of the error")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,dresidualGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,dresidualLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\nabla r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$", labelpad=10)
    plt.title("Derivative of the discrete residual")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(x,dresidualGD_out - derrorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,dresidualLSGD_out - derrorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\nabla\{r_M(u^{\boldsymbol{\alpha},\boldsymbol{\omega}}) - r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\}$", labelpad=10)
    plt.title("Derivative of the discretization error")
    auto_format_y_axis()
    figures.append(plt.gcf())
    plt.show()
    
    return figures

def save_and_plot_spectrum(nets, file=False):
    
    figures = []
    
    netINI, netGD, netLSGD = nets
    
    data = dict()
    
    if netINI != None:
        residualINI = residual(netINI)
        modesINI, spectrumINI_out = residualINI.spectrum()
        
        data['modes'] = modesINI
        data['coeffINI'] = spectrumINI_out
        data['accumINI'] = np.cumsum(spectrumINI_out)
    
    if netGD != None:
        residualGD = residual(netGD)
        modesGD, spectrumGD_out = residualGD.spectrum()
        
        data['modes'] = modesGD
        data['coeffGD'] = spectrumGD_out
        data['accumGD'] = np.cumsum(spectrumGD_out)
        
        
    if netGD != None:
        residualLSGD = residual(netLSGD)
        modesLSGD, spectrumLSGD_out = residualLSGD.spectrum()
        
        data['modes'] = modesLSGD
        data['coeffLSGD'] = spectrumLSGD_out
        data['accumLSGD'] = np.cumsum(spectrumLSGD_out)
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    plot_width, plot_height = 6.4, 4.8
    
            
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(modesINI,spectrumINI_out,'o', markersize=6, label="Initial",color="C5", alpha=0.5)
    plt.plot(modesGD,spectrumGD_out,marker='s', linestyle='None', markersize=3, label="Adam",color="C1")
    plt.plot(modesLSGD,spectrumLSGD_out,marker='^', linestyle='None', markersize=3, label="LS/Adam",color="C0")
    plt.ylabel(r"$\{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_m)-l(v_m)\}^2$", labelpad=10)
    plt.xlabel(r"$m$")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.axvline(x=M, color='black', linestyle='--', label=rf"cut-off $M={M}$")
    plt.legend()
    plt.title(r"Spectral coefficients of $r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(modesINI,data['accumINI'],'o', markersize=6, label="Initial",color="C5", alpha=0.5)
    plt.plot(modesGD,data['accumGD'],marker='s', linestyle='None', markersize=3, label="Adam",color="C1")
    plt.plot(modesLSGD,data['accumLSGD'],marker='^', linestyle='None', markersize=3, label="LS/Adam",color="C0")
    plt.ylabel(r"$\sum_{s=1}^m \{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_s)-l(v_s)\}^2$", labelpad=10)
    plt.xlabel(r"$m$")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.axvline(x=M, color='black', linestyle='--', label=rf"cut-off $M={M}$")
    plt.legend()
    plt.title(r"Accumulated spectral coefficients of $r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
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
    data['error2GD'] = history["error2GD"]
    data['error2LSGD'] = history["error2LSGD"]
    data['rel_errorGD'] = history["rel_errorGD"]
    data['rel_errorLSGD'] = history["rel_errorLSGD"]
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    plot_width, plot_height = 6.4, 4.8
            
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(iterations, history["lossGD"],label=r"$\mathcal{L}(\boldsymbol{\alpha},\boldsymbol{\omega})$ Adam", color="C1",lw=3, alpha=0.5)
    plt.plot(iterations, history["lossLSGD"],label=r"$\mathcal{L}(\boldsymbol{\alpha},\boldsymbol{\omega})$ LS/Adam", color="C0",lw=3, alpha=0.5)
    plt.plot(iterations, history["error2GD"],label=r"$\Vert e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{U}}^2$ Adam", color="red", linestyle="--", lw=1)
    plt.plot(iterations, history["error2LSGD"],label=r"$\Vert e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{U}}^2$ LS/Adam", color="purple", linestyle="--", lw=1)
    #plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$\mathcal{L}(\boldsymbol{\alpha},\boldsymbol{\omega}) \quad \text{and} \quad \Vert e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\Vert_{\mathbb{U}}^2$", labelpad=10)
    plt.xlabel("Iteration")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    plt.legend()
    plt.title("Training history")
    figures.append(plt.gcf())
    plt.show()
    
    return figures