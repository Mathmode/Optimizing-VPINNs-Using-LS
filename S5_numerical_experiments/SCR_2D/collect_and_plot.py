#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from config import LEGEND_LOCATION,EXACT,M1,M2
from SCR_2D.models import error, residual

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

                
# Define the formatter function
def thousands_formatter(x, pos):
    return f'{x:,.0f}'


def auto_format_colorbar(p, label=None):
    
    # get the colorbar object
    cbar = plt.colorbar(p)
    
    # Get the current colorbar limits
    c_min, c_max = p.get_clim()
    
    # Check if colorbar limits are in the form of small decimal values
    if abs(c_min) < 1 and abs(c_max) < 1 and c_min != c_max:
        exponent = int(np.floor(np.log10(max(abs(c_min), abs(c_max)))))
        
        # Only apply the formatting if the exponent is in the range we care about
        if exponent < 0:
            scale_factor = 10 ** -exponent
            
            # Set the formatter for the colorbar
            cbar.formatter = plt.FuncFormatter(lambda val, pos: f'{val * scale_factor:.1f}')
            cbar.update_ticks()
            
            # Add the scale label above the colorbar
            scale_label = r'$\times 10^{{{}}}$'.format(exponent)
            # originally, 0.5 and 1.05
            cbar.ax.text(0.7, 1.03, scale_label, transform=cbar.ax.transAxes, ha='center', va='bottom')
            
            # Add the additional label parallel to the right of the colorbar
    if label != None:
        cbar.set_label(label, labelpad=20)

def rescale_colorbar(c, colorbar_label):
    z_min, z_max = np.min(c.get_array()), np.max(c.get_array())
    exponent = 0
    scale_factor = 1.0

    # Calculate the range of values
    value_range = z_max - z_min
    
    # Check if the range is small enough to apply scaling
    if value_range < 1 and value_range > 0.1:
        exponent = int(np.floor(np.log10(value_range)))
        scale_factor = 10**(-exponent)
        c.set_array(c.get_array() * scale_factor)
    
    colorbar = plt.colorbar(c)
    
    if exponent != 0:
        scaled_label = f"$\\times 10^{{{exponent}}}$"
        colorbar.ax.text(1.1, 1.05, scaled_label, transform=colorbar.ax.transAxes, fontsize=15, va='center')
        
        # Customize the tick labels to show only one decimal place if rescaled
        colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    else:
        # Default tick format for integers and non-rescaled values
        colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x == int(x) else f'{x:.1f}'))
    
    colorbar.set_label(colorbar_label, labelpad=15)
    colorbar.update_ticks()
    
    return colorbar

class LogFormatterSciNotation(ticker.LogFormatter):
    def __call__(self, x, pos=None):
        return "$10^{{{:.0f}}}$".format(np.log10(x))


def save_and_plot_net(x, y, nets, file=False):
    
    figures = []
    
    netINI, netGD, netLSGD = nets

    exact_out = EXACT(x, y)
    
    data = {'x': x.numpy().flatten(),
            'y': y.numpy().flatten(),
            'exact': exact_out.numpy().flatten()
            }
    
    if netINI != None:
        errorINI = error(netINI)
        netINI_out = netINI([x,y])
        errorINI_out = errorINI([x,y])
        derrorINI_outX, derrorINI_outY = errorINI.d([x,y])
        
        data['netINI'] = netINI_out.numpy().flatten()
        data['errorINI'] = errorINI_out.numpy().flatten()
        data['derrorINI'] = np.sqrt(np.square(derrorINI_outX.numpy()).flatten() + np.square(derrorINI_outY.numpy()).flatten())
        
    
    if netGD != None:
        errorGD = error(netGD)
        netGD_out = netGD([x,y])
        errorGD_out = errorGD([x,y])
        derrorGD_outX, derrorGD_outY = errorGD.d([x,y])
        
        data['netGD'] = netGD_out.numpy().flatten()
        data['errorGD'] = errorGD_out.numpy().flatten()
        data['derrorGD'] = np.sqrt(np.square(derrorGD_outX.numpy()).flatten() + np.square(derrorGD_outY.numpy()).flatten())
        
        
    if netLSGD != None:
        errorLSGD = error(netLSGD)
        netLSGD_out = netLSGD([x,y])
        errorLSGD_out = errorLSGD([x,y])
        derrorLSGD_outX, derrorLSGD_outY = errorLSGD.d([x,y])
        
        data['netLSGD'] = netLSGD_out.numpy().flatten()
        data['errorLSGD'] = errorLSGD_out.numpy().flatten()
        data['derrorLSGD'] = np.sqrt(np.square(derrorLSGD_outX.numpy()).flatten() + np.square(derrorLSGD_outY.numpy()).flatten())
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    

    plot_width, plot_height = 6.4, 4.8
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=exact_out,cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$u^*$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Exact solution")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=netGD_out,cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Prediction (Adam)")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=netLSGD_out,cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$u^{\boldsymbol{\alpha},\boldsymbol{\omega}}$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Prediction (LS/Adam)")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=errorGD_out,cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Error (Adam)")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=errorLSGD_out,cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Error (LS/Adam)")
    figures.append(plt.gcf())
    plt.show()
    
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=data['derrorGD'],cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$\vert \; \nabla e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\; \vert$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Gradient of the error (Adam)")
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    p = plt.scatter(x,y,c=data['derrorLSGD'],cmap="viridis",alpha=0.7)
    auto_format_colorbar(p, r"$\vert \; \nabla e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})\; \vert$")
    plt.xlabel(r"$x$", labelpad=0)
    plt.ylabel(r"$y$", labelpad=15)
    plt.title("Gradient of the error (LS/Adam)")
    figures.append(plt.gcf())
    plt.show()
   
    return figures

def save_and_plot_spectrum(nets, file=False):
    
    figures = []
    
    netINI, netGD, netLSGD = nets
    
    data = dict()
    
    if netINI != None:
        residualINI = residual(netINI)
        modesINIX, modesINIY, spectrumINI_out = residualINI.spectrum()
        
        data['modesX'] = modesINIX
        data['modesY'] = modesINIY
        data['coeffINI'] = spectrumINI_out
        data['accumINI'] = np.cumsum(spectrumINI_out)
    
    if netGD != None:
        residualGD = residual(netGD)
        modesGDX, modesGDY, spectrumGD_out = residualGD.spectrum()
        
        data['modesX'] = modesGDX
        data['modesY'] = modesGDY
        data['coeffGD'] = spectrumGD_out
        data['accumGD'] = np.cumsum(spectrumGD_out)
        
        
    if netGD != None:
        residualLSGD = residual(netLSGD)
        modesLSGDX, modesLSGDY, spectrumLSGD_out = residualLSGD.spectrum()
        
        data['modesX'] = modesLSGDX
        data['modesY'] = modesLSGDY
        data['coeffLSGD'] = spectrumLSGD_out
        data['accumLSGD'] = np.cumsum(spectrumLSGD_out)
    
    
    if file:
        df = pd.DataFrame(data)
        df.to_csv(file, index=False)
    
    plot_width, plot_height = 6.4, 4.8
    
    plt.figure(figsize=(plot_width, plot_height))
    
    # Scatter plot with logarithmic color scaling
    p = plt.scatter(modesINIX, modesINIY, c=spectrumGD_out/spectrumINI_out, cmap="inferno", alpha=0.7, norm=LogNorm())
    
    # Colorbar with logarithmic ticks
    cbar = plt.colorbar(p, format=ticker.LogFormatter(labelOnlyBase=False))
    #cbar.set_label(r"$\{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_{m_1,m_2})-l(v_{m_1,m_2})\}^2$", labelpad=10)
    
    # Set custom ticks formatter for the colorbar

    
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    
    # Set logarithmic scaling for x and y axes
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    
    # Set labels for axes
    plt.xlabel(r"$m_1$\; ($x$-axis)", labelpad=10)
    plt.ylabel(r"$m_2$\; ($y$-axis)", labelpad=15)
    plt.axvline(x=M1, color='black', linestyle='--', label=rf'x-cutoff $M_1 = {M1}$')
    plt.axhline(y=M2, color='black', linestyle=':', label=rf'y-cutoff $M_2 = {M2}$')
    #plt.legend(loc="upper right")
    plt.title(r"Ratios of spectral coefficients: $\frac{Adam}{Initial}$",pad=10)
    figures.append(plt.gcf())
    plt.show()
    
    plt.figure(figsize=(plot_width, plot_height))
    
    # Scatter plot with logarithmic color scaling
    p = plt.scatter(modesINIX, modesINIY, c=spectrumLSGD_out/spectrumINI_out, cmap="inferno", alpha=0.7, norm=LogNorm())
    
    # Colorbar with logarithmic ticks
    cbar = plt.colorbar(p, format=ticker.LogFormatter(labelOnlyBase=False))
    #cbar.set_label(r"$\{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_{m_1,m_2})-l(v_{m_1,m_2})\}^2$", labelpad=10)
    
    # Set custom ticks formatter for the colorbar

    
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    
    # Set logarithmic scaling for x and y axes
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    
    # Set labels for axes
    plt.xlabel(r"$m_1$\; ($x$-axis)", labelpad=10)
    plt.ylabel(r"$m_2$\; ($y$-axis)", labelpad=15)
    plt.axvline(x=M1, color='black', linestyle='--', label=rf'x-cutoff $M_1 = {M1}$')
    plt.axhline(y=M2, color='black', linestyle=':', label=rf'y-cutoff $M_2 = {M2}$')
    #plt.legend(loc="upper right")
    plt.title(r"Ratios of spectral coefficients: $\frac{LS/Adam}{Initial}$",pad=10)
    figures.append(plt.gcf())
    plt.show()
    
            
    # plt.figure(figsize=(plot_width, plot_height))
    # p = plt.scatter(modesINIX,modesINIY,c=spectrumINI_out,cmap="inferno",alpha=0.7)
    # cbar = plt.colorbar(p, format='%.0e')
    # cbar.set_label(r"$\{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_{m_1,m_2})-l(v_{m_1,m_2})\}^2$", labelpad=10)
    # plt.xscale("log", base=2)
    # plt.xlabel(r"$m_1$", labelpad=0)
    # plt.yscale("log", base=2)
    # plt.ylabel(r"$m_2$",labelpad=15)

# Colorbar with logarithmic scale (automatic)

    
    # rescale_colorbar(p, r"$e(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    # plt.scatter(modesINIX,modesINIY,c=spectrumINI_out,cmap="viridis",alpha=0.7)
    # plt.plot(modesGD,spectrumGD_out,marker='s', linestyle='None', markersize=3, label="Adam",color="C1")
    # plt.plot(modesLSGD,spectrumLSGD_out,marker='^', linestyle='None', markersize=3, label="LS/Adam",color="C0")
    # plt.ylabel(r"$\{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_m)-l(v_m)\}^2$", labelpad=10)
    # plt.xlabel(r"$m$")
    # plt.xscale("log", base=2)
    # plt.yscale("log")
    # plt.axvline(x=M, color='black', linestyle='--', label=rf"cut-off $M={M}$")
    # plt.legend()
    # plt.title(r"Spectral coefficients of $r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    # figures.append(plt.gcf())
    # plt.show()
    
    # plt.figure(figsize=(plot_width, plot_height))
    # plt.plot(modesINI,data['accumINI'],'o', markersize=6, label="Initial",color="C5", alpha=0.5)
    # plt.plot(modesGD,data['accumGD'],marker='s', linestyle='None', markersize=3, label="Adam",color="C1")
    # plt.plot(modesLSGD,data['accumLSGD'],marker='^', linestyle='None', markersize=3, label="LS/Adam",color="C0")
    # plt.ylabel(r"$\sum_{s=1}^m \{b(u^{\boldsymbol{\alpha},\boldsymbol{\omega}},v_s)-l(v_s)\}^2$", labelpad=10)
    # plt.xlabel(r"$m$")
    # plt.xscale("log", base=2)
    # plt.yscale("log")
    # plt.axvline(x=M, color='black', linestyle='--', label=rf"cut-off $M={M}$")
    # plt.legend()
    # plt.title(r"Accumulated spectral coefficients of $r(u^{\boldsymbol{\alpha},\boldsymbol{\omega}})$")
    # figures.append(plt.gcf())
    # plt.show()
    
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