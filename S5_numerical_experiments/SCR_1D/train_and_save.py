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
from config import RESULTS_FOLDER, EXPERIMENT_REFERENCE, EPOCHS, XPLOT, KTEST
NAMES = ["training", "spectrum", "accumspectrum", "prediction", "error", "discreteresidual", "discretizationerror", "derror", "ddiscreteresidual", "ddiscretizationerror"]

x = XPLOT

netINI, netGD, netLSGD = u_net(), u_net(), u_net()
netINI(XPLOT), netGD(XPLOT), netLSGD(XPLOT)
netGD.set_weights(netINI.get_weights()), netLSGD.set_weights(netINI.get_weights())

print()
print("#######################")
print(" PLOTS BEFORE TRAINING ")
print("#######################")
print()
figures1 = save_and_plot_net(x,[netINI, netGD,netLSGD],file=False)

print()
print("########################")
print("        TRAINING        ")
print("########################")
print()
loss = loss_GDandLSGD(netGD, netLSGD)
loss.compile()
training = loss.fit(x, epochs=EPOCHS, batch_size=KTEST, callbacks=[EndOfTrainingLS()])

print()
print("########################")
print("  PLOTS AFTER TRAINING  ")
print("########################")
print()
figures2 = save_and_plot_loss(training.history,file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_training.csv")
figures3 = save_and_plot_spectrum([netINI, netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_spectrum.csv")
figures4 = save_and_plot_net(x,[netINI, netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_predictions.csv")

# We save all the figures (figures1, figures2 and figures3) in a pdf.
for i, fig in enumerate(figures2 + figures3 + figures4):
    NAME = NAMES[i]
    fig.savefig(f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_{NAME}.png", format='png', dpi=150, bbox_inches='tight')