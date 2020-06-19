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


% Description of data_organizer.m:
% This script organizes the data for the model fits in order for it to be
% passed to the GPU


%% Array and Cell Pre-allocation %%
% Pre-allocate the array that stores the number of predictors for each
% model fit
num_predictors_h = zeros(num_fits, 1);

% Pre-allocate the array that stores the number of observations for each
% model fit
num_observations_h = zeros(num_fits, 1);

% Pre-allocate the array that stores the indices that correspond to where
% the predictors for the model fits begin
B_thread_stride_h = zeros(num_fits, 1);

% Pre-allocate the array that stores the indices that correspond to where
% the observations for the model fits begin
observation_thread_stride_h = zeros(num_fits, 1);

% Pre-allocate the array that stores the indices that correspond to where
% the model matrices for the model fits begin
X_matrix_thread_stride_h = zeros(num_fits, 1);

% Pre-allocate the array that stores the intercept flags for the model
% fits (if the intercept flag is 0, then that means that the model matrix 
% does not include a column of ones corresponding to an intercept, and if
% it is 1, then that means that the model matrix does include a column of
% ones)
intercept_flag_h = zeros(num_fits, 1);

% Declare and initialize the variables that are used to calculate the
% indices that are stored into the pre-allocated arrays
predictor_count = 0;
observation_count = 0;
X_matrix_count = 0;

% Declare the cell array to store the model matrices for the model fits
X_cell = cell(1, num_fits);

% Declare the cell array to store the data to which the model matrices are
% being fit 
y_cell = cell(1, num_fits);


%% Obtain and Store the Data %%
% Obtain and store the data for each model fit
for ii = 1:num_fits
    % Specify the filename
    filename = ['model_data_' num2str(ii) '.mat'];
    
    % Load in the data for the model fit
    load(fullfile(data_path, filename));
    
    % Obtain the number of predictors and observations for the model fit
    num_predictors_h(ii) = size(X, 2);
    num_observations_h(ii) = size(X, 1);
    
    % Store the index for where the predictors for the model fit begin
    B_thread_stride_h(ii) = predictor_count;
    
    % Store the index for where the observations for the model fit begin
    observation_thread_stride_h(ii) = observation_count;
    
    % Store the index for where the model matrix for the model fit begins
    X_matrix_thread_stride_h(ii) = X_matrix_count;
    
    % Store the intercept flag for the model matrix
    intercept_flag_h(ii) = intercept_flag;
    
    % Store the model matrix for the model fit
    X_cell{ii} = reshape(X(:), [1, num_observations_h(ii) * num_predictors_h(ii)]);
    
    % Store the data to which the model matrix is being fit
    y_cell{ii} = reshape(y(:), [1, num_observations_h(ii)]);
    
    % Update the counts
    predictor_count = predictor_count + num_predictors_h(ii);
    observation_count = observation_count + num_observations_h(ii);
    X_matrix_count = X_matrix_count + (num_predictors_h(ii) .* num_observations_h(ii));
    
end

% Store the total number of predictors, observations, and model matrix
% values
total_num_B_values = predictor_count;
total_num_y_observations = observation_count;
total_num_X_matrix_values = X_matrix_count;

% Convert the cell array containing the model matrices into an array
X_matrix_h = cell2mat(X_cell);

% Convert the cell array containing the data to which the model matrices
% are being fit into an array
y_h = cell2mat(y_cell);


%% Clear Variables %%
% Clear the variables that are no longer needed
clear X_cell y_cell predictor_count observation_count X_matrix_count