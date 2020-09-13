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


% Description of test_GENRE_shared_memory.m:
% This function is used to test that the GENRE (GPU Elastic-Net REgression) 
% software package works on a system (it specifically tests the case when
% shared memory is used)


function test_GENRE_shared_memory()

% Clear all workspace variables and close all figures
clear all; close all; 


%% Define Parameters %%
fprintf('Testing beginning.\n');
% Create a new directory
mkdir('Test_Shared_Memory_Models');

% Define the number of model fits to perform
num_fits = 200;

% Obtain the current directory
current_directory = pwd;

% Define the path to the directory in which the model data files will be 
% saved
save_path = fullfile(current_directory, 'Test_Shared_Memory_Models');


%% Test Data Generation %%
fprintf('Beginning test data generation.\n');
% Generate and save the model data for each model fit
for ii = 1:num_fits
    % Randomly generate the number of observations and predictors for the 
    % model fit 
    num_observations = randi([50, 100], 1);
    num_predictors = randi([200, 300], 1);
    
    % Create the model matrix for the model fit
    X = randn(num_observations, num_predictors);
    
    % Add an intercept term to the model 
    intercept_flag = 1;
    
    % Add a column vector of ones to the beginning of the model matrix if 
    % an intercept term is supposed to be included
    if intercept_flag == 1
        X = [ones(num_observations, 1), X];
        num_predictors = num_predictors + 1;
    end
    
    % Randomly generate the model coefficients
    B = randn(num_predictors, 1) .* 100;
    
    % Create the observed data to which the model matrix will be fit
    y = X * B;
    
    % Define the name of the file to which the model data will be saved
    filename = ['model_data_' num2str(ii) '.mat'];
    
    % Save the data for the model fit
    save(fullfile(save_path, filename), '-v7.3', 'X', 'y', 'intercept_flag');

end
fprintf('Test data generation complete.\n');


%% Define Additional Parameters %%
% Specify single precision
precision = 'single';

% Specify the path to the files that contain the data for the model fits
data_path = save_path;

% Create a new directory
mkdir('Test_Shared_Memory_Model_Coefficients');

% Specify the path to save out the parameters and the computed model coefficients
% for the model fits
save_path = fullfile(current_directory, 'Test_Shared_Memory_Model_Coefficients');

% Specify the name of the output file to which the model coefficients
% computed using single precision are saved
output_filename = 'model_coefficients_single_precision.mat';

% Define or load in the alpha values that are used for the model fits 
alpha_values_h = rand(num_fits, 1); 

% Define or load in the lambda values that are used for the model fits 
lambda_values_h = repmat(0.001, [num_fits, 1]);

% Define the tolerance values that are used for the model fits 
tolerance_values_h = repmat(1E-4, [num_fits, 1]);

% Define the maximum iterations values that are used for the model fits 
max_iterations_values_h = repmat(100000, [num_fits, 1]);

% Specify the flag that determines which transformation option to use for 
% all of the model fits (1 = standardize the predictors on the GPU and 
% return the unstandardized model coefficients, 2 = normalize the
% predictors on the GPU and return the unnormalized model coefficients,
% 3 = the predictors are already standardized and return the standardized
% model coefficients, and 4 = the predictors are already normalized and
% return the normalized model coefficients). Note that the same 
% transformation flag has to be used for all of the model fits.
transformation_flag = 1;


%% Single Precision GENRE Code Test %%
fprintf('Single precision GENRE code test beginning.\n');
% Call the GENRE.m function to perform the model fits on the GPU using
% single precision
B_cell_single_precision = GENRE(precision, num_fits, data_path, save_path, output_filename, ...
    alpha_values_h, lambda_values_h, tolerance_values_h, max_iterations_values_h, ...
    transformation_flag);

% Check to see if all of the computed model coefficients are NaN or 0 (this 
% most likely indicates that the GPU processing was not successful)
B_single_precision = cell2mat(B_cell_single_precision);
if sum(isnan(B_single_precision)) == length(B_single_precision)
    error('All of the computed model coefficients were returned as NaN.');
end
if all(B_single_precision == 0)
    error('All of the computed model coefficients were returned as NaN');
end
fprintf('Single precision GENRE code test complete.\n');


%% Double Precision GENRE Code Test %%
% Specify double precision
precision = 'double';

% Specify the name of the output file to which the model coefficients
% computed using double precision are saved
output_filename = 'model_coefficients_double_precision.mat';

fprintf('Double precision GENRE code test beginning.\n');
% Call the GENRE.m function to perform the model fits on the GPU using
% double precision
B_cell_double_precision = GENRE(precision, num_fits, data_path, save_path, output_filename, ...
    alpha_values_h, lambda_values_h, tolerance_values_h, max_iterations_values_h, ...
    transformation_flag);

% Check to see if all of the computed model coefficients are NaN or 0 (this 
% most likely indicates that the GPU processing was not successful)
B_double_precision = cell2mat(B_cell_double_precision);
if sum(isnan(B_double_precision)) == length(B_double_precision)
    error('All of the computed model coefficients were returned as NaN.');
end
if all(B_double_precision == 0)
    error('All of the computed model coefficients were returned as NaN');
end
fprintf('Double precision GENRE code test complete.\n');
fprintf('Testing complete.\n');

end