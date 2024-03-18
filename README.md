# Island of Misfit Toys

![Island of Misfit Toys Banner](IslandOfMisfitToys.jpg)

Welcome to the **Island of Misfit Toys**! 

[Our PyPI page](https://pypi.org/project/IslandOfMisfitToys/) and [Documentation (work in progress)](https://islandofmisfittoys.readthedocs.io/en/latest/index.html). 

This repository contains implementations of various misfit functions found in seismic inversion literature. 
Our goal is to create a comprehensive collection of these functions to aid researchers and practitioners of seismic inverse problems.
We aim for a rapid prototyping environment built on top of [deepwave](https://ausargeo.com/deepwave/), the primary dependency of our package. 
Deepwave represents acoustic and elastic forward solvers as a pytorch module and thus integrates pytorch's optimization packages seamlessly into an FWI workflow.

If you have interest in adding features, please add a pull request or contact me directly at tyler@oden.utexas.edu. If you have any trouble with our code or find bugs, please let us know by filling out an issue!

For those curious about our logo's inspiration, search "Misfit Toys" in [Rudolph the Red-nosed Reindeer](https://en.wikipedia.org/wiki/Rudolph_the_Red-Nosed_Reindeer_(TV_special)), see the [Marmousi model](https://en.wikipedia.org/wiki/Marmousi_model), and the importance of misfits in [inverse problems](https://en.wikipedia.org/wiki/Inverse_problem#General_statement_of_the_inverse_problem).

## Table of Contents

- [Getting Started](#getting-started)
- [Misfit Functions (roadmap for future)](#roadmap)

## Getting Started

To begin using the misfit functions in this repository, simply run

```bash
pip install IslandOfMisfitToys
```

and should **should** be ready to go on any linux or arm64 MacOS system. Please let us know if you have trouble with the installation.

## Roadmap

(1) One-dimensional $W_1$ and $W_2$ (trace-by-trace) 
  - [Yang et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Application+of+optimal+transport+and+the+quadratic+Wasserstein+metric+to+full-waveform+inversion&btnG=)

(2) Huber
  - [Guitton and Symes 2003](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Guitton%2C+A.%2C+and+W.+W.+Symes%2C+2003%2C+Robust+inversion+of+seismic+data+using+the+Huber+norm%3A+Geophysics&btnG=)

(3) $L^1$-$L^2$ hybrid
  - [Bube and Langan](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Hybrid+l1%E2%88%95l2+minimization+with+applications+to+tomography&btnG=)

(4) Normalized and unnormalized Sobolev norms

  - [Zhu et al. 2021](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Implicit+regularization+effects+of+the+Sobolev+norms+in+image+processing&btnG=)

(5) Fisher-Rao metric

  - [Zhou et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=The+Wasserstein-Fisher-Rao+metric+for+waveform+based+earthquake+location&btnG=)

(6) Graph-space OT

  - [Metivier et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Optimal+transport+for+mitigating+cycle+skipping+in+full-waveform+inversion%3A+A+graph-space+transform+approach&btnG=)

(7) Entropic regularization OT

  - [Cuturi 2013](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Sinkhorn+distances%3A+Lightspeed+computation+of+optimal+transport&btnG=)

(8) Misfits based on reduced-order models

  - [Borcea et al. 2023](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Waveform+inversion+via+reduced+order+modeling+borcea&btnG=)
