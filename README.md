# GEN

# Overview
GEN (GPU Elastic-Net) is a MATLAB package that allows for many instances of linear regression with elastic-net regularization to be performed in parallel on a GPU.

# Setup
In order to utilize GEN, a CUDA-capable NVIDIA GPU along with an available release of MATLAB is required. The MATLAB Parallel Computing Toolbox must also be installed if not already installed in order to allow for the compilation of the MEX-files that contain CUDA code. Moreover, a C/C++ compiler that is compatible with the installed release of MATLAB must be installed in order to compile MEX-files containing C/C++ code.
