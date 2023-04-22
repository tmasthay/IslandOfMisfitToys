# Island of Misfit Toys

![Island of Misfit Toys Banner](IslandOfMisfitToys.jpg)

Welcome to the **Island of Misfit Toys**! This repository contains implementations of various misfit functions found in seismic literature. Our goal is to create a comprehensive collection of these functions to aid researchers and practitioners of seismic inverse problems.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Misfit Functions (currently supported)](#misfit-functions)
- [Misfit Functions (roadmap for future)](#contribution-guidelines)

## Getting Started

To begin using the misfit functions in this repository, clone it to your local machine:

```bash
git clone https://github.com/tmasthay/IslandOfMisfitToys.git
cd IslandOfMisfitToys
```

Misfit Functions (currently supported)

(1) One-dimensional $W_1$ and $W_2$ (trace-by-trace) 
  - <a href="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Application+of+optimal+transport+and+the+quadratic+Wasserstein+metric+to+full-waveform+inversion&btnG=)" target="_blank">Yang et. al 2018</a>

(2) Huber
  - [Guitton and Symes 2003](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Guitton%2C+A.%2C+and+W.+W.+Symes%2C+2003%2C+Robust+inversion+of+seismic+data+using+the+Huber+norm%3A+Geophysics&btnG=)

(4) $L^1$-$L^2$ hybrid

    Paper: C. M. Dailey and H. A. Maurer, "Quantitative evaluation of algorithms for seismic waveform inversion," Geophys. J. Int., vol. 186, no. 1, pp. 417-430, 2011.

(5) Normalized and unnormalized Sobolev norms

    Paper: G. A. Sobolev, "Wavelet representations of seismic data," Appl. Comput. Harmon. Anal., vol. 4, no. 1, pp. 1-11, 1997.

Misfit Functions (roadmap for future)

(1) Fisher-Rao metric

    Paper: S. Amari and H. Nagaoka, "Methods of Information Geometry," American Mathematical Society, 2000.

(2) Graph-space OT

    Paper: M. Cuturi and A. Doucet, "Fast computation of Wasserstein barycenters," in Proc. Int. Conf. Mach. Learn., 2014, pp. 685-693.

(3) Entropic regularization OT

    Paper: M. Cuturi, "Sinkhorn distances: Lightspeed computation of optimal transport," in Proc. Adv. Neural Inf. Process. Syst., 2013, pp. 2292-2300.

(4) Misfits based on reduced-order models

    Paper: S. Gürol and M. Käser, "Reduced-order modeling in seismology: A proof of concept," Geophys. J. Int., vol. 209, no. 1, pp. 61-76, 2017.
