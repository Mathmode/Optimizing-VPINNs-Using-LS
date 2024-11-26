# Optimizing Variational Physics-Informed Neural Networks Using Least Squares

Pieces of code for the experiments described in the manuscript **"Optimizing Variational Physics-Informed Neural Networks Using Least Squares"**, whose preprint is available at [https://arxiv.org/pdf/2407.20417](https://arxiv.org/pdf/2407.20417).

The code is organized into independent and self-contained sections as follows:

### Section `S3_automatic_differentiation`

Relates to Section 3 in the manuscript. To reproduce the manuscript results, execute `outerAD.py`.*

### Section `S4_GD_and_LSGD_optimizers`

Relates to Section 4 in the manuscript. To reproduce the manuscript results, execute `outerGDandLSGD.py`.*

### Section `S5_numerical_results`

Relates to Section 5 in the manuscript. To reproduce the manuscript results, execute `S5_all_experiments.py`.

We carried out the experiments in an **Ubuntu 22.04.4 LTS** system running **Linux kernel version 5.15.0-102-generic** on an **x86_64 architecture**, equipped with an **AMD EPYC 9474F 48-Core Processor** (2 threads per core) operating at a frequency of **4.11 GHz** with **377 GB of RAM**. 

*For measuring floating-point operations (FLOPs), we utilized the **'perf' profiler tool for Linux** [(https://perf.wiki.kernel.org)](https://perf.wiki.kernel.org) via the CPU-dependent command `perf stat -e fp_ret_sse_avx_ops.all [python-path] [python-script]`. We highlight that such a command depends on the machine's specifications.
