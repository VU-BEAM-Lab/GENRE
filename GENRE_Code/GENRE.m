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


% Description of GENRE.m:
% This is the main function that is used to call the GENRE (GPU Elastic-Net 
% REgression) software package from within MATLAB

% User-Defined Parameters:
% precision: Specifies whether to use single precision or double precision (there is
% typically a performance penalty when using double precision instead of
% single precision on GPUs, but using single precision has the trade-off of
% reduced numerical precision). Depending on factors such as the 
% conditioning of the model matrices, this reduced precision can lead to 
% significantly different results. Therefore, if you select single
% precision, then you should ensure that this precision is sufficient
% for your application. If you are uncertain, then it is recommended to use
% double precision.

% num_fits: Specifies the number of model fits

% data_path: Specifies the path to the files that contain the data for the 
% model fits

% save_path: Specifies the path to save out the parameters and the computed 
% model coefficients for the model fits

% output_filename: Specifies the name of the output file

% alpha_values_h: Vector containing the alpha values that are used for the
% model fits

% lambda_values_h: Vector containing the lambda values that are used for the
% model fits

% tolerance_values_h: Vector containing the tolerance values that are used
% for the model fits

% max_iterations_values_h: Vector containing the maximum iterations values
% that are used for the model fits

% transformation_flag: Specifies the flag that determines which 
% transformation option to use for all of the model fits (1 = standardize 
% the predictors on the GPU and return the unstandardized model 
% coefficients, 2 = normalize the predictors on the GPU and return the 
% unnormalized model coefficients, 3 = the predictors are already 
% standardized and return the standardized model coefficients, and 4 = the 
% predictors are already normalized and return the normalized model 
% coefficients). Note that the same transformation flag has to be used for 
% all of the model fits.

% Output:
% B_cell: Cell containing the computed model coefficients that are obtained 
% for each model fit (the model coefficients are also saved along with 
% other parameters using the specified save path and output filename)


function B_cell = GENRE(precision, num_fits, data_path, save_path, output_filename, ...
    alpha_values_h, lambda_values_h, tolerance_values_h, max_iterations_values_h, ...
    transformation_flag)

%% Clear MEX %%
% Clear MEX to unload any MEX-functions from memory
clear mex;

%% Data Organization %%
% Call the data_organizer.m script file to organize the data for the model
% fits
fprintf('Beginning data organization.\n');
data_organizer;
fprintf('Data organization complete.\n');

% Obtain the data type of the input data
input_data_type = class(X_matrix_h(1));


%% Parameter Check %%
% Make sure that all of the models include an intercept term if the
% transformation flag is set to 1
if transformation_flag == 1 && double(sum(intercept_flag_h)) ~= length(intercept_flag_h)
    error('All models must include an intercept term when transformation_flag is set to 1. This is not required for other values of transformation_flag.');
end


%% GPU Memory Estimation %%
fprintf('Beginning GPU memory estimation.\n');
GPU_memory_estimator;
fprintf('GPU memory estimation complete.\n');


%% GPU Thread Block Configuration Calculations %%
% Determine the maximum number of observations for one model fit across all
% of the model fits
max_num_observations = max(num_observations_h); 

% Determine whether GPU shared memory can be utilized for the model fits or
% not (if shared memory is used, then the code allocates no more than
% 32,000 bytes of shared memory per CUDA block)
shared_memory_per_block_max = 32000;
if gpu_properties.MaxShmemPerBlock >= shared_memory_per_block_max
    if strcmp(precision, 'single')
        max_threads_per_block = shared_memory_per_block_max ./ (max_num_observations .* 4);
        num_threads_per_block = pow2(floor(log2(max_threads_per_block)));
        if num_threads_per_block >= 32
            shared_memory_flag = 1;
            if num_threads_per_block > 128
                num_threads_per_block = 128;
            end
        else
            shared_memory_flag = 0;
            num_threads_per_block = 128;
        end
    elseif strcmp(precision, 'double')
        max_threads_per_block = shared_memory_per_block_max ./ (max_num_observations .* 8);
        num_threads_per_block = pow2(floor(log2(max_threads_per_block)));
        if num_threads_per_block >= 32
            shared_memory_flag = 1;
            if num_threads_per_block > 128
                num_threads_per_block = 128;
            end
        else
            shared_memory_flag = 0;
            num_threads_per_block = 128;
        end
    end
else
    shared_memory_flag = 0;
    num_threads_per_block = 128;
end


%% Call the C/CUDA Code to Perform the Model Fits %%
% Create the GPU parameters array
GPU_params_h = [transformation_flag, num_fits, total_num_y_observations, ...
    total_num_X_matrix_values, total_num_B_values, max_num_observations, shared_memory_flag, ...
    num_threads_per_block];

% Pre-allocate the cell array that stores the model coefficients for each
% fit
B_cell = cell(num_fits, 1);

% Determine which precision to use
if strcmp(precision, 'single')
    % Convert the precision of the input arrays
    GPU_params_h = double(GPU_params_h);
    observation_thread_stride_h = double(observation_thread_stride_h);
    num_observations_h = double(num_observations_h);
    num_predictors_h = double(num_predictors_h);
    X_matrix_thread_stride_h = double(X_matrix_thread_stride_h);
    y_h = single(y_h);
    X_matrix_h = single(X_matrix_h);
    B_thread_stride_h = double(B_thread_stride_h);
    intercept_flag_h = single(intercept_flag_h);
    alpha_values_h = single(alpha_values_h);
    lambda_values_h = single(lambda_values_h);
    tolerance_values_h = single(tolerance_values_h);
    max_iterations_values_h = single(max_iterations_values_h);
    
    % Call the MEX-file that uses single precision for the calculations  
    fprintf('Beginning GPU processing.\n');
    [B] = GENRE_GPU_single_precision(GPU_params_h, num_observations_h, ...
        observation_thread_stride_h, num_predictors_h, X_matrix_thread_stride_h, ...
        X_matrix_h, B_thread_stride_h, intercept_flag_h, alpha_values_h, lambda_values_h, tolerance_values_h, ...
        max_iterations_values_h, y_h);
    fprintf('GPU processing complete.\n');
    
    % Convert the model coefficients for each model fit to double precision
    % if the input data type is double
    if strcmp(input_data_type, 'double')
        B = double(B);
    end
    
    % Store the model coefficients for each model fit 
    start_ind = 1;
    for ii = 1:num_fits
        B_cell{ii} = B(start_ind:(start_ind + double(num_predictors_h(ii)) - 1));
        start_ind = start_ind + double(num_predictors_h(ii));
    end
elseif strcmp(precision, 'double')
    % Convert the precision of the input arrays
    GPU_params_h = double(GPU_params_h);
    observation_thread_stride_h = double(observation_thread_stride_h);
    num_observations_h = double(num_observations_h);
    num_predictors_h = double(num_predictors_h);
    X_matrix_thread_stride_h = double(X_matrix_thread_stride_h);
    y_h = double(y_h);
    X_matrix_h = double(X_matrix_h);
    B_thread_stride_h = double(B_thread_stride_h);
    intercept_flag_h = double(intercept_flag_h);
    alpha_values_h = double(alpha_values_h);
    lambda_values_h = double(lambda_values_h);
    tolerance_values_h = double(tolerance_values_h);
    max_iterations_values_h = double(max_iterations_values_h);
       
    % Call the MEX-file that uses double precision for the calculations
    fprintf('Beginning GPU processing.\n');
    [B] = GENRE_GPU_double_precision(GPU_params_h, num_observations_h, ...
        observation_thread_stride_h, num_predictors_h, X_matrix_thread_stride_h, ...
        X_matrix_h, B_thread_stride_h, intercept_flag_h, alpha_values_h, lambda_values_h, tolerance_values_h, ...
        max_iterations_values_h, y_h); 
    fprintf('GPU processing complete.\n');
    
     % Convert the model coefficients for each model fit to single precision
    % if the input data type is single
    if strcmp(input_data_type, 'single')
        B = single(B);
    end
    
    % Store the model coefficients for each model fit 
    start_ind = 1;
    for ii = 1:num_fits
        B_cell{ii} = B(start_ind:(start_ind + num_predictors_h(ii) - 1));
        start_ind = start_ind + num_predictors_h(ii);
    end   
end


%% Save the Parameters and Model Coefficients %%
fprintf('Beginning saving of outputs.\n');
save(fullfile(save_path, output_filename), '-v7.3', 'B_cell', 'precision', ...
    'alpha_values_h', 'lambda_values_h', 'tolerance_values_h', ...
    'max_iterations_values_h', 'transformation_flag');
fprintf('Saving of outputs complete.\n');

end