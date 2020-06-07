# GEN

# Overview
GEN (GPU Elastic-Net) is a MATLAB package that allows for many instances of linear regression with elastic-net regularization to be performed in parallel on a GPU.

# Setup
In order to utilize GEN, a CUDA-capable NVIDIA GPU along with an available release of MATLAB is required. The MATLAB Parallel Computing Toolbox must also be installed if not already installed in order to allow for the compilation of MEX-files containing CUDA code. Moreover, a C/C++ compiler that is compatible with the installed release of MATLAB must be installed in order to compile MEX-files containing C/C++ code. The compiler compatibility can be found at https://www.mathworks.com/support/requirements/supported-compilers.html. Note that the code was evaluated on both Windows 10 and Ubuntu 16.04 LTS. For Windows, the free community edition of Microsoft Visual Studio 2017 was used as the C/C++ compiler. To download this older version, go to https://visualstudio.microsoft.com/vs/older-downloads/ and create a free Dev Essentials program account with Microsoft. When installing Microsoft Visual Studio 2017, make sure to also check the box for the VC++ 2015 toolset (the 2015 will most likely be followed by a version number). For Ubuntu, 
