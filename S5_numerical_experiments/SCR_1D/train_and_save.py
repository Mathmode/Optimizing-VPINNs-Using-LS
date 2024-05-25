#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
# Load all the functions and classes from SCR_1D AFTER creating the 'config.py' script
from SCR_1D.models import u_net
from SCR_1D.loss import loss_GDandLSGD
from SCR_1D.collect_and_plot import save_and_plot_net, save_and_plot_spectrum, save_and_plot_loss
from SCR_1D.callbacks import EndOfTrainingLS
from config import RESULTS_FOLDER, EXPERIMENT_REFERENCE, EPOCHS, XPLOT

x = XPLOT

netGD, netLSGD = u_net(), u_net()
netGD(x), netLSGD(x)
netLSGD.set_weights(netGD.get_weights())

# We activate this backend to save all the plt's in 'save_and_plot_net' 
# in a pdf file
from matplotlib.backends.backend_pdf import PdfPages

print()
print("#######################")
print(" PLOTS BEFORE TRAINING ")
print("#######################")
print()
figures1 = save_and_plot_net(x,[netGD,netLSGD],file=False)
figures2 = save_and_plot_spectrum([netGD,netLSGD],file=False)

print()
print("########################")
print("        TRAINING        ")
print("########################")
print()
loss = loss_GDandLSGD(netGD, netLSGD)
loss.compile()
training = loss.fit(x, epochs=EPOCHS, batch_size=1000, callbacks=[EndOfTrainingLS()])

print()
print("########################")
print("  PLOTS AFTER TRAINING  ")
print("########################")
print()
figures3 = save_and_plot_loss(training.history,file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_training.csv")
figures4 = save_and_plot_spectrum([netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_spectrum.csv")
figures5 = save_and_plot_net(x,[netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_predictions.csv")


# We save all the figures (figures1, figures2 and figures3) in a pdf.
with PdfPages(f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_figures.pdf") as pdf:
    for fig in figures1 + figures2 + figures3 + figures4 + figures5:
        pdf.savefig(fig)