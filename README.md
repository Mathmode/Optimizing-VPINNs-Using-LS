# Optimizing VPINNs Using LS

Pieces of codes of the manuscript "Optimizing Variational Physics-Informed Neural Networks Using Least Squares" are available in [link to the preprint soon!]

Section S3_automatic_differentiation relates to Section 3 in the manuscript. To reproduce the manuscript results, execute 'outer.py'.

Section S4_LSGD_and_GD_optimizers relates to Section 4 in the manuscript. To reproduce the manuscript results, execute 'outerGDandLSGD.py'.

Section S5_numerical_results relates to Section 5 in the manuscript. To reproduce the manuscript results, execute 'S5_all_experiments.py'.

We originally* carried out the experimentation in an Ubuntu 22.04.4 LTS system running Linux kernel version 5.15.0-102-generic on an x86_64 architecture, equipped with an AMD EPYC 9474F 48-Core Processor (2 threads per core) operating at a frequency of 4.11 GHz with 377 GB of RAM. For measuring FLOPs, we utilized the 'perf' profiler tool for Linux (https://perf.wiki.kernel.org, Last accessed: April 2024) via the CPU-dependent command 'perf stat -e fp_ret_sse_avx_ops.all [python-path] [python-script].

*Last run: July 2024.
