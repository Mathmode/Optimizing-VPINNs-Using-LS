#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on May, 2024

@author: curiarteb
"""
# Load all the functions and classes from SCR_1D AFTER creating the 'config.py' script
import tensorflow as tf
from SCR_2D.models import u_net
from SCR_2D.loss import loss_GDandLSGD
from SCR_2D.collect_and_plot import save_and_plot_net, save_and_plot_spectrum, save_and_plot_loss
from SCR_2D.callbacks import EndOfTrainingLS
from config import RESULTS_FOLDER, EXPERIMENT_REFERENCE, EPOCHS, XYPLOT

x0,y0 = XYPLOT

# For the Cartesian product
x = tf.repeat(x0, y0.shape[0], axis=0)
y = tf.tile(y0, (x0.shape[0], 1))

netINI, netGD, netLSGD = u_net(), u_net(), u_net()
netINI([x,y]), netGD([x,y]), netLSGD([x,y])
netGD.set_weights(netINI.get_weights()), netLSGD.set_weights(netINI.get_weights())

# We activate this backend to save all the plt's in 'save_and_plot_net' 
# in a pdf file
from matplotlib.backends.backend_pdf import PdfPages

print()
print("#######################")
print(" PLOTS BEFORE TRAINING ")
print("#######################")
print()
figures1 = save_and_plot_net(x,y,[netINI,netGD,netLSGD],file=False)

print()
print("########################")
print("        TRAINING        ")
print("########################")
print()
loss = loss_GDandLSGD(netGD, netLSGD)
loss.compile()
training = loss.fit(x,y,epochs=EPOCHS, batch_size=x.shape[0], callbacks=[EndOfTrainingLS()])

print()
print("########################")
print("  PLOTS AFTER TRAINING  ")
print("########################")
print()
figures2 = save_and_plot_loss(training.history,file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_training.csv")
figures3 = save_and_plot_spectrum([netINI,netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_spectrum.csv")
figures4 = save_and_plot_net(x,y,[netINI,netGD,netLSGD],file=f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_predictions.csv")


# We save all the figures in a pdf.
with PdfPages(f"{RESULTS_FOLDER}/{EXPERIMENT_REFERENCE}_figures.pdf") as pdf:
    for fig in figures1 + figures2 + figures3 + figures4:
        pdf.savefig(fig, bbox_inches='tight')