---
title: 'GENRE (GPU Elastic-Net REgression): A CUDA-Accelerated Package for Massively Parallel Linear Regression with Elastic-Net Regularization'
tags:
  - CUDA
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
GENRE (GPU Elastic-Net REgression) is a package that allows for many instances of linear
regression with elastic-net regularization to be processed in parallel on a GPU by using the C programming 
language and NVIDIA's (NVIDIA Corporation, Santa Clara, CA, USA) Compute Unified Device Architecture (CUDA) parallel
programming framework. Linear regression with elastic-net regularization [@zou_hastie_2005] is a widely 
utilized tool when performing model-based analyses. The basis of this method is that it allows for a 
combination of L1-regularization and L2-regularization to be applied to a given regression problem. 
Therefore, feature selection and coefficient shrinkage are performed while still allowing for the 
presence of groups of correlated features. The process of performing these model fits can be 
computationally expensive, and one of the fastest packages that is currently available is glmnet 
[@friedman_hastie_tibshirani_2010; @qian_hastie_friedman_tibshirani_simon_2013; @hastie_qian_2014]. 
This package provides highly efficient Fortran implementations of several different types of regression. 
In the case of its implementation of linear regression with elastic-net regularization, the objective 
function shown in (eq. \ref{eq:1}) is minimized. 

\begin{equation}
\boldsymbol{\hat\beta} = \underset{\boldsymbol{\beta}}{\mathrm{argmin}}\frac{1}{2N}\sum_{i=1}^{N}  \left(\boldsymbol{y}_{i} - \sum_{j=1}^{P} \boldsymbol{X}_{ij}\boldsymbol{\beta}_{j}\right)^{2} + \lambda \left( \alpha \left\| \boldsymbol{\beta} \right\|_{1} + \frac{ \left(1 - \alpha \right)\left\| \boldsymbol{\beta} \right\|_{2}^{2}}{2} \right) \label{eq:1}
\end{equation}

To minimize this objective function, cyclic coordinate descent is utilized as the optimization algorithm.
This algorithm consists of minimizing the objective function with respect to one model coefficient at a time.
Cycling through all of the coefficients results in one iteration, and this process continues until specified
convergence criteria are satisfied. As previously stated, glmnet is highly efficient for single model fits, but
performing thousands of these fits will still require significant computational time due to each one being executed 
in a serial fashion on a CPU. However, by using GENRE, massively parallel processing can be performed in order to 
achieve a significant speedup. This is due to the fact that modern GPUs consist of thousands of computational cores
that can be utilized. Moreover, although the processing in GENRE is performed using the C programming language and 
CUDA, a MEX-interface is included to allow for this code to be called within the MATLAB (The MathWorks, Inc., Natick, MA, USA) 
programming language for convenience. This also means that with modification, the MEX-interface can be replaced with 
another interface if it is desired to call the C/CUDA code in another language, or the C/CUDA code can be utilized 
without an interface. Note that other packages have been developed that can utilize GPUs for linear regression with
elastic-net regularization, such as H2O4GPU [@H2O4GPU_2020]. However, for this application, these packages typically 
focus on performing parallel computations on the GPU for one model fit at a time in order to achieve acceleration when 
compared to a serial CPU implementation. For GENRE, the computations for a single model fit are not parallelized on the 
GPU. Instead, many model fits on the GPU are executed in parallel, where each model fit is performed by one computational thread.

# Statement of Need
The core motivation for developing GENRE was that many of the available packages for performing linear regression with elastic-net regularization focus on achieving high performance in terms of computational time or resource consumption for single model fits. However, they often do not address the case in which there is a need to perform many model fits in parallel. For example, the research project that laid the foundation for GENRE involved performing ultrasound image reconstruction using an algorithm called Aperture Domain Model Image REconstruction (ADMIRE) [@byram_jakovljevic_2014; @byram_dei_tierney_dumont_2015; @dei_byram_2017]. This algorithm is computationally expensive due to the fact that in one stage, it requires thousands of instances of linear regression with elastic-net regularization to be performed in order to fit models of ultrasound data. When this algorithm was implemented on a CPU, it typically required an amount of time that was on the scale of minutes to reconstruct one ultrasound image. The primary bottleneck was performing all of the required model fits due to the fact that a custom C implementation of cyclic coordinate descent was used to compute each fit serially. However, a GPU implementation of the algorithm was developed, and this implementation provided a speedup of over two orders of magnitude, which allowed for multiple ultrasound images to be reconstructed per second. For example, on a computer containing dual Intel (Intel Corporation, Santa Clara, CA) Xeon Silver 4114 CPUs @ 2.20 GHz with 10 cores each along with an NVIDIA GeForce GTX 1080 Ti GPU and an NVIDIA GeForce RTX 2080 Ti GPU, the CPU implementation of ADMIRE had an average processing time of 90.033 $\pm$ 0.494 seconds for one frame of ultrasound channel data while the GPU implementation had an average processing time of 0.433 $\pm$ 0.001 seconds. The average processing time was obtained for each case by taking the average of 10 runs for the same dataset, and timing was performed using MATLAB's built-in timing capabilities. The 2080 Ti GPU was used to perform GPU processing, and the number of processing threads was set to 1 for the CPU implementation. The main contributor to this speedup was the fact that the model fits were performed in parallel on the GPU. For this particular case, 152,832 model fits were performed. Note that double precision was used for the CPU implementation while single precision was utilized for the GPU implementation due to the fact there is typically a performance penalty when using double precision on a GPU. Moreover, for the CPU implementation, MATLAB was used, and a MEX-file was used to call the C implementation of cyclic coordinate descent for the model fitting stage. In addition, note that one additional optimization when performing the model fits on the GPU in the case of ADMIRE is that groups of model fits can use the same model matrix, which allows for improved coalesced memory access and GPU memory bandwidth use. This particular optimization is not used by GENRE.

Aside from this application, there are a number of other applications that can potentially benefit from having the ability to perform model fits in a massively parallel fashion, which is why the code was developed into a package. For example, linear regression with elastic-net regularization has been applied to the field of genomics in order to develop predictive models that utilize genetic markers [@ogutu_schulz-streeck_piepho_2012; @waldmann_mészáros_gredler_fuerst_sölkner_2013]. In addition, like ADMIRE, there are a variety of other signal processing applications. For example, this regression method has been used to create models of functional magnetic resonance imaging data in order to predict the mental states of subjects and provide insight into neural activity [@carroll_cecchi_rish_garg_rao_2009]. Moreover, another signal processing example is that linear regression models with elastic-net regularization have been used in combination with hidden Markov random field segmentation to perform computed tomography estimation for the purposes of magnetic resonance imaging-based attenuation correction for positron emission tomography/magnetic resonance imaging [@chen_juttukonda_lee_su_espinoza_lin_shen_lulash_an_2014]. Now, by using GENRE, the models in each of the aforementioned examples can be computed in parallel in order to reduce the amount of processing time that is required.

# Example Benchmark Comparing GENRE with glmnet
GENRE has the potential to provide significant speedup due to the fact that many model fits can be performed in parallel on a GPU. Therefore, an example benchmark was performed where we compared GENRE with glmnet, which is written in Fortran and performs the model fits in a serial fashion on a CPU. In this benchmark, 20,000 model matrices were randomly generated within MATLAB. Each model matrix consisted of 50 observations and 200 predictors (50x200), and an intercept term was included for all of the models. Note that to add an intercept term in GENRE, a column of ones was appended at the beginning of each model matrix to make the predictor dimension 201 (adding a column of ones is not required for glmnet). For each model matrix, the model coefficients were randomly generated, and the matrix multiplication of the model matrix and the coefficients was performed to obtain the observation vector. Therefore, this provided 20,000 observation vectors with each containing 50 observations. Once the data was generated, both GENRE and glmnet were used to perform the model fits and return the computed model coefficients. An $\alpha$ value of 0.5 and a $\lambda$ value of 0.001 were used for all of the model fits. The tolerance convergence criterion for both packages was set to 1E-4. It was also specified for each package to standardize the model matrices, which means that the unstandardized model coefficients were returned. Note that the column of ones for each model matrix corresponding to the intercept term is not standardized in the case of GENRE. GENRE allows for the user to select either single precision or double precision for performing the model fits on the GPU, so processing was done for both cases. The MATLAB version of the glmnet software package includes a compiled executable MEX-file that allows for Fortran code to be called, and it uses double precision for the calculations. In addition, due to the fact that all of the model matrices have a small number of observations (50) in this case, GENRE is also able to use shared memory in addition to global memory when performing the model fits. Shared memory has lower latency than global memory, so utilizing it can provide performance benefits. Therefore, processing was performed both with and without using shared memory. 

The computer that was used for the benchmarks contained dual Intel Xeon Silver 4114 CPUs @ 2.20 GHz with 10 cores each along with an NVIDIA GeForce GTX 1080 Ti GPU and an NVIDIA GeForce RTX 2080 Ti GPU. The 2080 Ti GPU was used to perform GPU processing. For each case, the average of 10 runs was taken, and timing was performed using MATLAB's built-in timing capabilities. Note that GENRE has a data organization step that loads the data for the model fits from files and organizes it into the format that is used by the GPU. For this benchmark, this step was not counted in the timing due to the fact that it was assumed that all of the data was already loaded into MATLAB on the host system for both GENRE and glmnet. The GPU times include the time it takes to transfer data for the model fits from the host system to the GPU, standardize the model matrices, perform the model fits, unstandardize the model coefficients, transfer the computed model coefficients back from the GPU to the host system, and store the coefficients into a MATLAB cell structure. The CPU time includes the time it takes to standardize the model matrices, perform the model fits, unstandardize the model coefficients, and store the coefficients into a MATLAB cell structure. The benchmark results are shown in Table \ref{tab:table1} below. Note that DP, SP, and SMEM correspond to double precision, single precision, and shared memory respectively. In addition, note that the input data for the model fits was of type `double` for this benchmark. Therefore, in the case of GENRE, some of the inputs would need to be converted to type `single` before they are passed to the GPU when using single precision for the computations. Moreover, GENRE also converts the datatype of the computed model coefficients to the datatype of the original input data. This means that for the single precision cases, the computed model coefficients would need to be converted to be type `double` after they are passed back to the host system from the GPU. For purposes of benchmarking the single precision cases, the time to perform the type conversions of the inputs to type `single` was not included, and the returned model coefficients were just kept as type `single`. This is due to the fact that including these times would increase the benchmark times for the single precision cases in this scenario, and if it were a different scenario, the double precision cases could be impacted instead of the single precision cases. For example, if the type of the original input data was `single` and double precision was used for the calculations, then these datatype conversions would have to be made for the double precision cases, but they would not have to be made for the single precision cases.

\vspace{0.3 cm}

 \begin{centering}
 \begin{table}[h!]
 \caption{\label{tab:table1}Benchmark times (seconds).}
 \footnotesize
\begin{tabular}[h]{| c | c | c | c | c |} 
\hline
\textbf{GENRE DP} & \textbf{GENRE DP SMEM} & \textbf{GENRE SP} & \textbf{GENRE SP SMEM} & \textbf{glmnet} \\
\hline
 1.367 $\pm$ 0.024 & 1.137 $\pm$ 0.015 & 1.048 $\pm$ 0.008 & 0.909 $\pm$ 0.006 & 10.319 $\pm$ 0.039 \\
\hline
\end{tabular}
\end{table}
\end{centering}

\vspace{0.3 cm}

As shown in Table \ref{tab:table1}, GENRE provides an order of magnitude speedup when compared to glmnet, and the best performance was achieved by using single precision with shared memory. For glmnet, the benchmark result that is shown was obtained by using the naive algorithm option for the package because this option was faster than the covariance algorithm option. For example, the benchmark result that was obtained when using the covariance algorithm option was 32.546 $\pm$ 0.201 seconds. In addition, it is important to note that in these benchmarks, most of the time for GENRE was spent transferring the model matrices from the host system to the GPU. However, there are cases when once the model matrices have been used in one call, they can be reused in subsequent calls. For example, a user might want to reuse the same model matrices except just change the $\alpha$ value or the $\lambda$ value that is used in elastic-net regularization, or they might want to just change the observation vectors that the model matrices are fit to. By default, each time GENRE is called, the \textit{clear mex} command is executed, and the GENRE MEX-files are setup so that all allocated memory on the GPU is freed when this command is called. However, in a case where the model matrices can be reused after they are transferred once, the \textit{clear mex} command can be removed. Essentially, every time one of the MEX-files for GENRE is called for the first time, all of the data for the model fits will be transferred to the GPU. However, if the \textit{clear mex} command is removed, then for subsequent calls, all of the data for the model fits will still be transferred except for the model matrices, which will be kept on the GPU from the first call. By not having to transfer the model matrices again, performance can be significantly increased. To demonstrate this, the same benchmark from above was repeated, but for each case this time, GENRE was called once before performing the 10 runs. This is to replicate the case where the model matrices are reused in subsequent calls. The benchmark results are shown in Table \ref{tab:table2} below. Note that the benchmark for glmnet was not repeated.

\vspace{0.3 cm}

 \begin{centering}
  \begin{table}[h!]
 \caption{\label{tab:table2}Benchmark times (seconds).}
  \footnotesize
\begin{tabular}[h]{| c | c | c | c | c |} 
\hline
\textbf{GENRE DP} & \textbf{GENRE DP SMEM} & \textbf{GENRE SP} & \textbf{GENRE SP SMEM} & \textbf{glmnet} \\
\hline
 0.307 $\pm$ 0.003 & 0.083 $\pm$ 0.001 & 0.197 $\pm$ 0.002 & 0.056 $\pm$ 0.001 & 10.319 $\pm$ 0.039 \\
\hline
\end{tabular}
\end{table}
\end{centering}

\vspace{0.3 cm}

As shown in Table \ref{tab:table2}, when the model matrices can be reused and do not have to be transferred again, GENRE provides a speedup of over two orders of magnitude when compared with glmnet, and using single precision with shared memory provides the best performance. This type of performance gain would most likely be difficult to achieve even when using a multi-CPU implementation of cyclic coordinate descent on a single host system. In addition, it is important to note that this benchmark was just to illustrate an example of when using GENRE provides performance benefits, but whether or not performance benefits are achieved depends on the problem. For example, in GENRE, one computational thread on the GPU is used to perform each model fit. Therefore, when many model fits need to be performed, the parallelism of the GPU can be utilized. However, if only one model fit needs to be performed, then using a serial CPU implementation such as glmnet will most likely provide better performance than GENRE due to factors such as CPU cores having higher clock rates and more resources per core than GPU cores.

# Acknowledgements
This work was supported by NIH grants R01EB020040 and S10OD016216-01 and NAVSEA grant N0002419C4302.

# References
