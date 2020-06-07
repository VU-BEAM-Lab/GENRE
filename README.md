# GEN (GPU Elastic-Net): A MATLAB Package for Massively Parallel Linear Regression with Elastic-Net Regularization

## Table of Contents
1. [Overview](#Overview)
2. [Setup](#Setup)
3. [Interface](#Interface)

## Overview
GEN (GPU Elastic-Net) is a MATLAB package that allows for many instances of linear regression with elastic-net regularization to be performed in parallel on a GPU. The specific onjective function that is minimized is shown below.

\begin{equation} \boldsymbol{\hat\beta} = \underset{\boldsymbol{\beta}}{\mathrm{argmin}}\frac{1}{2N}\sum_{i=1}^{N} \left(\boldsymbol{y}{i} - \sum{j=1}^{P} \boldsymbol{X}{ij}\boldsymbol{\beta}{j}\right)^{2} + \lambda \left( \alpha \left| \boldsymbol{\beta} \right|{1} + \frac{ \left(1 - \alpha \right)\left| \boldsymbol{\beta} \right|{2}^{2}}{2} \right) \label{eq:1} \end{equation}

## Setup
In order to utilize GEN, a CUDA-capable NVIDIA GPU along with an available release of MATLAB is required. The MATLAB Parallel Computing Toolbox must also be installed if not already installed in order to allow for the compilation of MEX-files containing CUDA code. Moreover, a C/C++ compiler that is compatible with the installed release of MATLAB must be installed in order to compile MEX-files containing C/C++ code. The compiler compatibility can be found at https://www.mathworks.com/support/requirements/supported-compilers.html. Note that the code was evaluated on both Windows and Linux OS. For Windows, the free community edition of Microsoft Visual Studio 2017 was used as the C/C++ compiler. To download this older version, go to https://visualstudio.microsoft.com/vs/older-downloads/ and create a free Dev Essentials program account with Microsoft. When installing Microsoft Visual Studio 2017, make sure to also check the box for the VC++ 2015 toolset (the 2015 will most likely be followed by a version number). For Linux, the GNU Compiler Collection (GCC) was used as the C/C++ compiler. In addition to a C/C++ compiler, a CUDA toolkit version that is compatible with the installed release of MATLAB must be installed. To determine compatibility, refer to https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html. Once the compatibility is determined, go to https://developer.nvidia.com/cuda-toolkit-archive and install the particular CUDA toolkit version. Note that the installation process for the toolkit will also allow for the option to install a new graphics driver. If you do not desire to install a new driver, then you must ensure that your current driver supports the toolkit version that is being installed. For driver and toolkit compatability, refer to page 4 of https://docs.nvidia.com/pdf/CUDA_Compatibility.pdf.

Before compiling the code, you should first check to see that MATLAB recognizes your GPU card. To do so, go to the command prompt and type ```gpuDevice```. If successful, the properties of the GPU will be displayed. If an error is returned, then possible causes will most likely be related to the graphics driver or the toolkit version that is installed. Once the GPU is recognized, the next step is to compile the MEX-files that contain the C/CUDA code. Assuming the code repository is already on your system, go to the MATLAB directory that contains the repository folders and add them to your MATLAB path. For Windows OS, type the following commands into the MATLAB command prompt.

```Matlab
cd GEN_GPU_Single_Precision_Code
mexcuda GEN_GPU_single_precision.cu
cd ..\GEN_GPU_Double_Precision_Code
mexcuda GEN_GPU_double_precision.cu
```

The same commands can be used for Linux OS, but the path to the CUDA toolkit library must also be included. This is illustrated by the following commands.

```Matlab
cd GEN_GPU_Single_Precision_Code
mexcuda GEN_GPU_single_precision.cu -L/usr/local/cuda-10.0/lib64
cd ../GEN_GPU_Double_Precision_Code
mexcuda GEN_GPU_double_precision.cu -L/usr/local/cuda-10.0/lib64
```

Note that there might be differences in your path compared to the one shown above, such as in regards to the version of the CUDA toolkit that is being used. In addition, if desired, the ```-v``` flag can be included at the end of each mexcuda command to display compilation details. If the compilation process is successful, then it will display a success message for each compilation in the command prompt. In addition, a compiled MEX-file will appear in each folder. The compilation is process is important, and it is recommended to recompile any time a different release of MATLAB is utilized.

## Interface
