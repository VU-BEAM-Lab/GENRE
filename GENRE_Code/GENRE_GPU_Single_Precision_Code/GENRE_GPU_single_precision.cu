// Copyright 2020 Christopher Khan

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the license at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and 
// limitations under the License.
        
        
// Description of GENRE_GPU_single_precision.cu: 
// This file contains the MEX-interface that calls the C/CUDA code for 
// performing the computations for GENRE using single precision
        
        
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "GPU_kernels_single_precision.cu"
#include "mex.h"
#define DEVICE_ID 0

// Declare the parameters
static int initialized = 0;              // Specifies whether everything has been initialized or not
static int transformation_flag;          // Flag that determines which predictor transformation option to use 
static int transformed = 0;              // Specifies whether the predictors have been transformed or not if it is selected for the predictors to be transformed on the GPU      
static int num_fits;                     // Total number of model fits that are performed
static int total_num_y_observations;     // Total number of values in the y_d array
static int total_num_X_matrix_values;    // Total number of values in the X_matrix_d array
static int total_num_B_values;           // Total number of values in the B_d array
static int max_num_observations;         // The maximum number of observations for one model fit across all of the model fits
static int shared_memory_flag;           // Flag that determines whether to use GPU shared memory or not
static int num_threads_per_block;        // Determines how many threads to use per block
        
// Declare the pointers to the GPU device arrays
static float * y_d;                            // Stores the datasets that are being fit with models
static float * residual_y_d;                   // Stores the residual values that are obtained during each fit
static float * y_std_d;                        // Stores the standard deviations for each portion of the y_d array (one portion corresponds to one fit)
static float * standardized_lambda_values_d;   // Stores the standardized lambda values for each portion of the y_d array (one portion corresponds to one fit)
static double * num_observations_d;            // Stores the number of observations for each fit 
static double * observation_thread_stride_d;   // Stores the indices corresponding to where the data for each fit starts in the y_d array (these indices use zero-based indexing)
static double * num_predictors_d;              // Stores the number of predictors for each fit 
static float * X_matrix_d;                     // Stores all of the model matrices
static double * X_matrix_thread_stride_d;      // Stores the indices corresponding to where each model begins in the X_matrix_d array (these indices use zero-based indexing)
static float * B_d;                            // Stores the predictor coefficient values that are obtained from each fit
static double * B_thread_stride_d;             // Stores the indices corresponding to where each set of predictor coefficients begins in the B_d array (these indices use zero-based indexing)
static float * alpha_values_d;                 // Stores the alpha values that are used in elastic-net regularization         
static float * tolerance_values_d;             // Stores the tolerance values that are used for cyclic coordinate descent
static float * max_iterations_values_d;        // Stores the maximum number of iterations values that are used for cyclic coordinate descent
static float * intercept_flag_d;               // Stores the flag that determines whether each model includes a column of ones corresponding to an intercept term or not
static float * scaling_factors_d;              // Stores the normalization or standardization factor for each predictor in each model matrix
static float * mean_X_matrix_d;                // Stores the mean of each predictor in each model matrix
static float * model_fit_flag_d;               // Stores the flag that determines whether to perform a model fit or not 
    


// Define the function that frees allocated memory on the GPU when the MEX-interface is exited
void cleanup() {
    
// Set the current GPU device
cudaSetDevice(DEVICE_ID);

// Print to the console
mexPrintf("MEX-file is terminating, destroying the arrays.\n");

// Free the GPU device arrays
cudaFree(y_d);
cudaFree(residual_y_d);
cudaFree(y_std_d);
cudaFree(num_observations_d);
cudaFree(observation_thread_stride_d);
cudaFree(num_predictors_d);
cudaFree(X_matrix_thread_stride_d);
cudaFree(X_matrix_d);
cudaFree(B_d);
cudaFree(B_thread_stride_d);
cudaFree(alpha_values_d);            
cudaFree(standardized_lambda_values_d);
cudaFree(tolerance_values_d);
cudaFree(max_iterations_values_d);
cudaFree(intercept_flag_d);
cudaFree(scaling_factors_d);
cudaFree(mean_X_matrix_d);
cudaFree(model_fit_flag_d);

// Reset the GPU device (need this for profiling the MEX-file using the Nvidia Visual Profiler)
cudaDeviceReset();

}

// Define the MEX gateway function
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {

// Initialize everything if it is the first call to the MEX-file
if (!initialized) {
    
   // Set the current GPU device
   cudaSetDevice(DEVICE_ID);

   // Print to the console
   mexPrintf("MEX-file initializing.\n");

   // Declare the pointers to the host arrays
   double * GPU_params_h;
   double * num_observations_h;
   double * observation_thread_stride_h;
   double * num_predictors_h;
   double * X_matrix_thread_stride_h;
   float * X_matrix_h;
   double * B_thread_stride_h;
   float * intercept_flag_h;
    
   // Obtain the array that contains the GPU parameters
   GPU_params_h = (double*)mxGetData(prhs[0]);
   transformation_flag = (int)GPU_params_h[0];
   num_fits = (int)GPU_params_h[1];
   total_num_y_observations = (int)GPU_params_h[2];
   total_num_X_matrix_values = (int)GPU_params_h[3];
   total_num_B_values = (int)GPU_params_h[4];
   max_num_observations = (int)GPU_params_h[5];
   shared_memory_flag = (int)GPU_params_h[6];
   num_threads_per_block = (int)GPU_params_h[7];

   // Obtain the other input arrays
   num_observations_h = (double*)mxGetData(prhs[1]);
   observation_thread_stride_h = (double*)mxGetData(prhs[2]);
   num_predictors_h = (double*)mxGetData(prhs[3]);
   X_matrix_thread_stride_h = (double*)mxGetData(prhs[4]);
   X_matrix_h = (float*)mxGetData(prhs[5]);
   B_thread_stride_h = (double*)mxGetData(prhs[6]);
   intercept_flag_h = (float*)mxGetData(prhs[7]);

   // Allocate the GPU device arrays
   cudaMalloc(&y_d, total_num_y_observations * sizeof(float));
   cudaMalloc(&residual_y_d, total_num_y_observations * sizeof(float));
   cudaMalloc(&y_std_d, num_fits * sizeof(float));
   cudaMalloc(&num_observations_d, num_fits * sizeof(double));
   cudaMalloc(&observation_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&num_predictors_d, num_fits * sizeof(double));
   cudaMalloc(&X_matrix_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&X_matrix_d, total_num_X_matrix_values * sizeof(float));
   cudaMalloc(&B_d, total_num_B_values * sizeof(float));
   cudaMalloc(&B_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&alpha_values_d, num_fits * sizeof(float));
   cudaMalloc(&standardized_lambda_values_d, num_fits * sizeof(float));
   cudaMalloc(&tolerance_values_d, num_fits * sizeof(float));
   cudaMalloc(&max_iterations_values_d, num_fits * sizeof(float));
   cudaMalloc(&intercept_flag_d, num_fits * sizeof(float));
   cudaMalloc(&scaling_factors_d, total_num_B_values * sizeof(float));
   cudaMalloc(&mean_X_matrix_d, total_num_B_values * sizeof(float));
   cudaMalloc(&model_fit_flag_d, num_fits * sizeof(float));

   // Transfer the data from the host arrays to the GPU device arrays
   cudaMemcpy(num_observations_d, num_observations_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(observation_thread_stride_d, observation_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(num_predictors_d, num_predictors_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(X_matrix_thread_stride_d, X_matrix_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(X_matrix_d, X_matrix_h, total_num_X_matrix_values * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_thread_stride_d, B_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(intercept_flag_d, intercept_flag_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
           
   // Run the cleanup function when exiting the MEX interface
   mexAtExit(cleanup);

   // Set initialization variable to 1 because everything has been initialized
   initialized = 1;

}

// Set the current GPU device
cudaSetDevice(DEVICE_ID);

// Declare the pointers to the host arrays
float * alpha_values_h;
float * lambda_values_h;
float * tolerance_values_h;
float * max_iterations_values_h;

// Obtain the parameter arrays
alpha_values_h = (float*)mxGetData(prhs[8]);
lambda_values_h = (float*)mxGetData(prhs[9]);
tolerance_values_h = (float*)mxGetData(prhs[10]);
max_iterations_values_h = (float*)mxGetData(prhs[11]);

// Transfer the data from the host arrays to the GPU device arrays
cudaMemcpy(alpha_values_d, alpha_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(standardized_lambda_values_d, lambda_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(tolerance_values_d, tolerance_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(max_iterations_values_d, max_iterations_values_h, num_fits * sizeof(float), cudaMemcpyHostToDevice);
        
// Declare the pointer to the host array
float * y_h;

// Obtain the input data array
y_h = (float*)mxGetData(prhs[12]);

// Transfer the input data from the host array to the GPU device array
cudaMemcpy(y_d, y_h, total_num_y_observations * sizeof(float), cudaMemcpyHostToDevice);

// Set the predictor coefficient values to 0
cudaMemset(B_d, 0, total_num_B_values * sizeof(float));

// Set the model fit flags to 0
cudaMemset(model_fit_flag_d, 0, num_fits * sizeof(float));


// Set num_threads_per_block to num_fits if the total number of model fits is less than the number of model fits per GPU block
if (num_fits < num_threads_per_block) {
   num_threads_per_block = num_fits;
}

// Calculate the number of GPU blocks that are required to perform all of the model fits
int num_blocks = (int)ceilf((float)num_fits / (float)num_threads_per_block);

// Calculate the number of model fits that are performed within the last GPU block
int num_threads_last_block = num_fits - ((num_blocks - 1) * num_threads_per_block);

// Perform predictor standardization or normalization if specified
if (transformation_flag == 1 && transformed == 0) {
   // Define the grid and block dimensions for the predictor_standardization GPU kernel
   dim3 PREDICTOR_STANDARDIZATION_GRID_SIZE;
   PREDICTOR_STANDARDIZATION_GRID_SIZE = dim3(num_blocks, 1, 1);
   dim3 PREDICTOR_STANDARDIZATION_BLOCK_SIZE;
   PREDICTOR_STANDARDIZATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);

   // Call the predictor_standardization GPU kernel in order to standardize the predictors of the model matrices
   predictor_standardization<<<PREDICTOR_STANDARDIZATION_GRID_SIZE, PREDICTOR_STANDARDIZATION_BLOCK_SIZE>>>(X_matrix_d, scaling_factors_d, mean_X_matrix_d, X_matrix_thread_stride_d, B_thread_stride_d, num_observations_d, num_predictors_d, intercept_flag_d, num_threads_per_block, num_blocks, num_threads_last_block);

   // Update the transformed variable
   transformed = transformed + 1;
} else if (transformation_flag == 2 && transformed == 0) {
   // Define the grid and block dimensions for the predictor_normalization GPU kernel
   dim3 PREDICTOR_NORMALIZATION_GRID_SIZE;
   PREDICTOR_NORMALIZATION_GRID_SIZE = dim3(num_blocks, 1, 1);
   dim3 PREDICTOR_NORMALIZATION_BLOCK_SIZE;
   PREDICTOR_NORMALIZATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);

   // Call the predictor_normalization GPU kernel in order to normalize the predictors of the model matrices
   predictor_normalization<<<PREDICTOR_NORMALIZATION_GRID_SIZE, PREDICTOR_NORMALIZATION_BLOCK_SIZE>>>(X_matrix_d, scaling_factors_d, X_matrix_thread_stride_d, B_thread_stride_d, num_observations_d, num_predictors_d, intercept_flag_d, num_threads_per_block, num_blocks, num_threads_last_block);

   // Update the transformed variable
   transformed = transformed + 1;
}

// Define the grid and block dimensions for the model_fit_preparation GPU kernel
dim3 MODEL_FIT_PREPARATION_GRID_SIZE;
MODEL_FIT_PREPARATION_GRID_SIZE = dim3(num_blocks, 1, 1);
dim3 MODEL_FIT_PREPARATION_BLOCK_SIZE;
MODEL_FIT_PREPARATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);  

// Call the model_fit_preparation GPU kernel in order to standardize the y data and the lambda values
model_fit_preparation<<<MODEL_FIT_PREPARATION_GRID_SIZE, MODEL_FIT_PREPARATION_BLOCK_SIZE>>>(y_d, residual_y_d, model_fit_flag_d, y_std_d, standardized_lambda_values_d, num_observations_d, observation_thread_stride_d, num_threads_per_block, num_blocks, num_threads_last_block);

// Define the grid and block dimensions for the model_fit_reconstruction GPU kernel
dim3 MODEL_FIT_GRID_SIZE;
MODEL_FIT_GRID_SIZE = dim3(num_blocks, 1, 1);
dim3 MODEL_FIT_BLOCK_SIZE;
MODEL_FIT_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);

// Call the model_fit GPU kernel in order to fit the models to the y data     
if (shared_memory_flag == 1) {
   model_fit_shared_memory<<<MODEL_FIT_GRID_SIZE, MODEL_FIT_BLOCK_SIZE, num_threads_per_block * max_num_observations * sizeof(float)>>>(B_d, B_thread_stride_d, model_fit_flag_d, X_matrix_d, X_matrix_thread_stride_d, observation_thread_stride_d, residual_y_d, y_std_d, standardized_lambda_values_d, num_observations_d, num_predictors_d, alpha_values_d, tolerance_values_d, max_iterations_values_d, intercept_flag_d, transformation_flag, num_threads_per_block, num_blocks, num_threads_last_block);
} else {
   model_fit<<<MODEL_FIT_GRID_SIZE, MODEL_FIT_BLOCK_SIZE>>>(B_d, B_thread_stride_d, model_fit_flag_d, X_matrix_d, X_matrix_thread_stride_d, observation_thread_stride_d, residual_y_d, y_std_d, standardized_lambda_values_d, num_observations_d, num_predictors_d, alpha_values_d, tolerance_values_d, max_iterations_values_d, intercept_flag_d, transformation_flag, num_threads_per_block, num_blocks, num_threads_last_block);
}

// Unstandardize or unnormalize the model coefficients if specified
if (transformation_flag == 1) {
   dim3 PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_GRID_SIZE;
   PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_GRID_SIZE = dim3(num_blocks, 1, 1);
   dim3 PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_BLOCK_SIZE;
   PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);   
   predictor_coefficient_unstandardization<<<PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_GRID_SIZE, PREDICTOR_COEFFICIENT_UNSTANDARDIZATION_BLOCK_SIZE>>>(B_d, B_thread_stride_d, model_fit_flag_d, X_matrix_d, X_matrix_thread_stride_d, scaling_factors_d, mean_X_matrix_d, num_predictors_d, intercept_flag_d, num_threads_per_block, num_blocks, num_threads_last_block);
} else if (transformation_flag == 2) {
   dim3 PREDICTOR_COEFFICIENT_UNNORMALIZATION_GRID_SIZE;
   PREDICTOR_COEFFICIENT_UNNORMALIZATION_GRID_SIZE = dim3(num_blocks, 1, 1);
   dim3 PREDICTOR_COEFFICIENT_UNNORMALIZATION_BLOCK_SIZE;
   PREDICTOR_COEFFICIENT_UNNORMALIZATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1); 
   predictor_coefficient_unnormalization<<<PREDICTOR_COEFFICIENT_UNNORMALIZATION_GRID_SIZE, PREDICTOR_COEFFICIENT_UNNORMALIZATION_BLOCK_SIZE>>>(B_d, B_thread_stride_d, model_fit_flag_d, X_matrix_d, X_matrix_thread_stride_d, scaling_factors_d, num_predictors_d, intercept_flag_d, num_threads_per_block, num_blocks, num_threads_last_block);
}

// Declare the pointer to the MEX-file output
float * B_h;

// Assign the pointer to the output array
plhs[0] = mxCreateNumericMatrix(total_num_B_values, 1, mxSINGLE_CLASS, mxREAL);
B_h = (float*)mxGetData(plhs[0]);

// Transfer the output
cudaMemcpy(B_h, B_d, total_num_B_values * sizeof(float), cudaMemcpyDeviceToHost);

}
