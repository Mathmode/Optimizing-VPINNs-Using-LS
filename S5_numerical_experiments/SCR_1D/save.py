#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

from config import EXACT,DEXACT, SOURCE
import pandas as pd

def save_and_plot_net(x, nets, file=False):
    
    figures = []
    
    netGD, netLSGD = nets
    
    exact_out = EXACT(x)
    dexact_out = DEXACT(x)
    
    data = {'x': x.numpy().flatten(),
            'exact': exact_out.numpy().flatten(),
            'dexact': dexact_out.numpy().flatten()
            }
    
    if netGD != None:
        netGD_out = netGD(x)
        dnetGD_out = netGD.dbwd(x)
        residualGD_out = -netGD.ddfwd(x) - SOURCE(x)
        errorGD_out = netGD_out - exact_out
        derrorGD_out = dnetGD_out - dexact_out
        
        data['netGD'] = netGD_out.numpy().flatten()
        data['dnetGD'] = dnetGD_out.numpy().flatten()
        data['residualGD'] = residualGD_out.numpy().flatten()
        data['errorGD']: errorGD_out.numpy().flatten()
        data['derrorGD']: derrorGD_out.numpy().flatten()
        
        
    if netLSGD != None:
        netLSGD_out = netLSGD(x)
        dnetLSGD_out = netLSGD.dbwd(x)
        residualLSGD_out = -netLSGD.ddfwd(x) - SOURCE(x)
        errorLSGD_out = netLSGD_out - exact_out
        derrorLSGD_out = dnetLSGD_out - dexact_out
        
        data['netLSGD'] = netLSGD_out.numpy().flatten()
        data['dnetLSGD'] = dnetLSGD_out.numpy().flatten()
        data['residualLSGD'] = residualLSGD_out.numpy().flatten()
        data['errorLSGD']: errorLSGD_out.numpy().flatten()
        data['derrorLSGD']: derrorLSGD_out.numpy().flatten()
    
    
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
    
    
    plt.plot(x,exact_out,label="exact",color="C2",lw=4)
    plt.plot(x,netGD_out,label="Adam", color="C1",lw=2)
    plt.plot(x,netLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.title(r"$u^{\alpha,\omega}(x)$ vs. $u^*(x)$")
    figures.append(plt.gcf())
    plt.show()


    plt.plot(x,errorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,errorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.title(r"$u^{\alpha,\omega}(x) - u^*(x)$")
    figures.append(plt.gcf())
    plt.show()
    
    plt.plot(x,residualGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,residualLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.title(r"$-(u^{\alpha,\omega})''(x) - f(x)$")
    figures.append(plt.gcf())
    plt.show()
    
    plt.plot(x,derrorGD_out,label="Adam",color="C1",lw=2)
    plt.plot(x,derrorLSGD_out,label="LS/Adam",color="C0",lw=1)
    plt.legend()
    plt.title(r"$\nabla u^{\alpha,\omega}(x) - \nabla u^*(x)$")
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
    
    plt.plot(iterations, history["lossGD"],label="loss Adam", color="C1",lw=2)
    plt.plot(iterations, history["lossLSGD"],label="loss LS/Adam", color="C0",lw=2)
    plt.plot(iterations, history["errorGD"],label="error Adam", color="red", linestyle="--", lw=1)
    plt.plot(iterations, history["errorLSGD"],label="error LS/Adam", color="purple", linestyle="--", lw=1)
    plt.yscale("log")
    plt.legend()
    plt.title(r"$Loss = \mathcal{L}(\alpha,\omega)$ and $error = \Vert u^{\alpha,\omega} - u^*\Vert$ history")
    figures.append(plt.gcf())
    plt.show()
    
    return figures