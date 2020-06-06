---
title: 'GEN: GPU Elastic-Net'
tags:
  - MATLAB
  - GPU computing
  - Cyclic coordinate descent
  - Elastic-net regularization
  - Linear regression
authors:
  - name: Christopher Khan
    orcid: 0000-0003-3201-3423
    affiliation: 1
  - name: Brett Byram
    orcid: 0000-0003-3693-1459
    affiliation: 1
affiliations: 
  - name: Vanderbilt University
    index: 1
date: 6 June 2020
bibliography: paper.bib
---

# Summary
GEN (GPU Elastic-Net) is a CUDA-accelerated MATLAB package that allows for many instances of linear
regression with elastic-net regularization to be processed in parallel on a GPU. Linear regression with 
elastic-net regularization [@zou_hastie_2005] is a widely utilized tool when performing model-based analyses. 
The basis of this method is that it allows for a combination of L1-regularization and L2-regularization
to be applied to a given regression problem. Therefore, feature selection and coefficient
shrinkage are performed while still allowing for the presence of groups of correlated features.
Now, the process of performing these model fits can be computationally expensive, and one of the 
fastest packages that is currently available is glmnet [@friedman_hastie_tibshirani_2010], 
[@qian_hastie_friedman_tibshirani_simon_2013], [@hastie_qian_2014]. This package provides highly efficient 
Fortran implementations of several different types of regression. In the case of its implementation of linear
regression with elastic-net regularization, the objective function shown in (eq. \ref{eq:1}) is minimized. 

\begin{equation}
\begin{aligned}
\boldsymbol{\hat\beta} = & \argmin\limits_{\boldsymbol{\beta}} \frac{1}{2N}\sum_{i=1}^{N}  \left(\boldsymbol{y}_{i} - \sum_{j=1}^{P} \boldsymbol{X}_{ij}\boldsymbol{\beta}_{j}\right)^{2} + \lambda \left( \alpha \left\| \boldsymbol{\beta} \right\|_{1} + \frac{ \left(1 - \alpha \right)\left\| \boldsymbol{\beta} \right\|_{2}^{2}}{2} \right) \label{eq:1} 
\end{aligned}
\end{equation}

To minimize this objective function, cyclic coordinate descent is utilized as the optimization algorithm.
This algorithm consists of minimizing the objective function with respect to one model coefficient at a time.
Cycling through all of the coefficients results in one iteration, and this process continues until specified
convergence criteria are satisfied. As previously stated, glmnet is highly efficient for single model fits, but
performing thousands of these fits will still require significant computational time due to each one being executed 
in a serial fashion on a CPU. However, by using GEN, massively parallel processing can be performed in order to 
achieve significant speedup. This is due to the fact that modern GPUs consist of thousands of computational cores
that can utilized. Moreover, although the processing in GEN is performed using the C programming language and 
the CUDA parallel programming framework, a MEX-interface is used to allow for this code to be called within the MATLAB 
programming language.

# Acknowledgements
This work was supported by NIH grants R01EB020040 and S10OD016216-01 and NAVSEA grant N0002419C4302.

# References
