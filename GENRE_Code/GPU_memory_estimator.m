% Copyright 2020 Christopher Khan

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the license at

%     http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and 
% limitations under the License.


% Description of GPU_memory_estimator.m:
% This script estimates the amount of GPU memory that is required to
% perform the model fits and ensures that the estimated amount does not
% exceed the amount of memory that is available on the GPU


%% Determine Precision %%
% Determine if single precision or double precision is being used for the
% GPU calculations
if strcmp(precision, 'single')
    num_bytes = 4;
elseif strcmp(precision, 'double')
    num_bytes = 8;
end


%% Array Memory Size Calculation %%
% Calculate the size in bytes of each array that is allocated on the GPU
y_d_size = total_num_y_observations * num_bytes;
residual_y_d_size = total_num_y_observations * num_bytes;
y_std_d_size = num_fits * num_bytes;
num_observations_d_size = num_fits * 8; % This array always uses double precision
observation_thread_stride_d_size = num_fits * 8; % This array always uses double precision
num_predictors_d_size = num_fits * 8; % This array always uses double precision
X_matrix_thread_stride_d_size = num_fits * 8; % This array always uses double precision
X_matrix_d_size = total_num_X_matrix_values * num_bytes;
B_d_size = total_num_B_values * num_bytes;
B_thread_stride_d_size = num_fits * 8; % This array always uses double precision
alpha_values_d_size = num_fits * num_bytes;
standardized_lambda_values_d_size = num_fits * num_bytes;
tolerance_values_d_size = num_fits * num_bytes;
max_iterations_values_d_size = num_fits * num_bytes;
intercept_flag_d_size = num_fits * num_bytes;
scaling_factors_d_size = total_num_B_values * num_bytes;
mean_X_matrix_d_size = total_num_B_values * num_bytes;
model_fit_flag_d_size = num_fits * num_bytes;


%% Estimate Required GPU Memory %%
% Calculate the estimate of GPU memory that is required to perform the
% model fits
estimated_GPU_memory_required_GB = (y_d_size + residual_y_d_size + y_std_d_size + num_observations_d_size ...
    + observation_thread_stride_d_size + num_predictors_d_size + X_matrix_thread_stride_d_size ...
    + X_matrix_d_size + B_d_size + B_thread_stride_d_size + alpha_values_d_size ...
    + standardized_lambda_values_d_size + tolerance_values_d_size + max_iterations_values_d_size ...
    + intercept_flag_d_size + scaling_factors_d_size + mean_X_matrix_d_size ...
    + model_fit_flag_d_size) ./ 1E9;


%% Print Estimate of Required GPU Memory %%
% Print the estimate of GPU memory that is required to perform the model fits
fprintf('Estimated GPU memory required: %f GB\n', estimated_GPU_memory_required_GB);


%% Compare Estimate of Required GPU Memory to Available GPU Memory %%
% This if statement makes sure that the estimate of GPU memory that is
% required to perform the model fits does not exceed the total amount of
% available memory on the GPU
gpu_properties = gpuDevice;
if estimated_GPU_memory_required_GB > (gpu_properties.AvailableMemory ./ 1E9) 
    error('Estimate of required GPU memory exceeds available GPU memory.');
end