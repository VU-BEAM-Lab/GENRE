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
        

// Description of GPU_kernels_single_precision.cu: 
// This file contains the CUDA code that allows for performing the computations
// for GENRE on a GPU using single precision
        
        
// Define the GPU kernel that performs predictor normalization
__global__ void predictor_normalization(float * X_matrix_d, float * scaling_factors_d, double * X_matrix_thread_stride_d, double * B_thread_stride_d, double * num_observations_d, double * num_predictors_d, float * intercept_flag_d, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) {
   // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
   int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];

   // Obtain the number of observations for the fit
   int num_observations = (int)num_observations_d[fit_ind];

   // Obtain the number of predictors for the fit
   int num_predictors = (int)num_predictors_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct model matrix for the fit
   int X_thread_stride = (int)X_matrix_thread_stride_d[fit_ind];

   // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
   int intercept_flag = (int)intercept_flag_d[fit_ind];

   // Declare and initialize the variable that stores the number of the first predictor column to be normalized
   int start_ind = 0;

   // This if statement makes sure to not normalize the first predictor column if it corresponds to the intercept term
   if (intercept_flag == 1) {
      start_ind = 1;
   }

   // Normalize each predictor column so that the sum of the square of each predictor column is equal to 1
   for (int predictor_column = start_ind; predictor_column < num_predictors; predictor_column++) {
       // Declare and initialize the variable that stores the sum of the square of the predictor column
       float sum_squared = 0.0f;

       // Calculate the sum of the square of the predictor column
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           float X_value = X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row];
           sum_squared = sum_squared + (X_value * X_value);
       }

       // Calculate the square root of the sum of the square of the predictor column
       float square_root_sum_squared = sqrtf(sum_squared);

       // Store the square root of the sum of the square of the predictor column
       scaling_factors_d[predictor_thread_stride + predictor_column] = square_root_sum_squared;

       // Normalize the predictor column by dividing each observation in the predictor column by the square root of the sum of the square of the predictor column
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] = X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] / square_root_sum_squared;
       }
   }

   // This if statement stores a scaling factor of 1 for the predictor column if it corresponds to an intercept term
   if (intercept_flag == 1) {
      scaling_factors_d[predictor_thread_stride] = 1.0f;
   }
}

}



// Define the GPU kernel that performs predictor standardization
__global__ void predictor_standardization(float * X_matrix_d, float * scaling_factors_d, float * mean_X_matrix_d, double * X_matrix_thread_stride_d, double * B_thread_stride_d, double * num_observations_d, double * num_predictors_d, float * intercept_flag_d, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) {
   // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
   int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];

   // Obtain the number of observations for the fit
   int num_observations = (int)num_observations_d[fit_ind];

   // Obtain the number of predictors for the fit
   int num_predictors = (int)num_predictors_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct model matrix for the fit
   int X_thread_stride = (int)X_matrix_thread_stride_d[fit_ind];

   // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
   int intercept_flag = (int)intercept_flag_d[fit_ind];

   // Declare and initialize the variable that stores the number of the first predictor column to be standardized
   int start_ind = 0;

   // This if statement makes sure to not standardize the first predictor column if it corresponds to the intercept term
   if (intercept_flag == 1) {
      start_ind = 1;
   }

   // Standardize each predictor column by subtracting the mean of the predictor column from each observation and diving each observation by the standard deviation of the predictor column
   for (int predictor_column = start_ind; predictor_column < num_predictors; predictor_column++) {
       // Declare and initialize the variable that stores the sum of the predictor column
       float sum_value = 0.0f;

       // Calculate the sum of the predictor column
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           float X_value = X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row];
           sum_value = sum_value + X_value;
       }

       // Calculate the mean of the predictor column
       float mean_value = sum_value / (float)num_observations;

       // Store the mean of the predictor column
       mean_X_matrix_d[predictor_thread_stride + predictor_column] = mean_value;

       // Declare and initialize the variable that stores the sum of the square of the demeaned predictor column
       float sum_squared = 0.0f;

       // Normalize the predictor column by dividing each observation in the predictor column by the square root of the sum of the square of the predictor column
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           float X_value_demeaned = X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] - mean_value;
           sum_squared = sum_squared + (X_value_demeaned * X_value_demeaned);
       }

       // Calculate the standard deviation of the demeaned predictor column
       float std = sqrtf(sum_squared / (float)num_observations);

       // Store the standard deviation of the demeaned predictor column
       scaling_factors_d[predictor_thread_stride + predictor_column] = std;

       // Standardize the predictor column by subtracting its mean and dividing by its standard deviation
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] = (X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] - mean_value) / std; 
       }
   }

   // This if statement stores a scaling factor of 1 and a mean of 1 for the first column if it corresponds to an intercept term
   if (intercept_flag == 1) {
      scaling_factors_d[predictor_thread_stride] = 1.0f;
      mean_X_matrix_d[predictor_thread_stride] = 1.0f;
   }
}

}



// Define the GPU kernel that calculates the standard deviations for each portion of the y_d array, standardizes the y_d array, and calculates the standardized lambda values
__global__ void model_fit_preparation(float * y_d, float * residual_y_d, float * model_fit_flag_d, float * y_std_d, float * standardized_lambda_values_d, double * num_observations_d, double * observation_thread_stride_d, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) {
   // Obtain the number of observations for the fit
   int num_observations = (int)num_observations_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct set of observations in the cropped_y_d array for the fit
   int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

   // Declare and initialize the variable that stores the running sum of y for the fit
   float sum_value = 0.0f;

   // Calculate the running sums for sum_value
   for (int observation = 0; observation < num_observations; observation++) {
       float value = y_d[observation_thread_stride + observation];
       sum_value += value;
   }

   // Calculate the mean of y for the fit
   float mean = sum_value / (float)num_observations;
   
   // Declare and initialize the variable that stores the standard deviation of y for the fit
   float std = 0.0f;

   // Calculate the standard deviation of y for the fit
   for (int observation = 0; observation < num_observations; observation++) {
       float value_2 = y_d[observation_thread_stride + observation];
       std += ((value_2 - mean) * (value_2 - mean));
   }
   std = sqrtf(std / (float)num_observations);

   // Store the standard deviation of y for the fit in the y_std_d array
   y_std_d[fit_ind] = std;        

   // This if statement standardizes the lambda values and the y data if the standard deviation isn't 0
   if (std != 0.0f) {
       // Set the model fit flag to 1 if the standard deviation is not 0 and a model fit should be performed
       model_fit_flag_d[fit_ind] = 1.0f;

       // Calculate the standardized lambda value and store it into the standardized_lambda_d array 
       standardized_lambda_values_d[fit_ind] = standardized_lambda_values_d[fit_ind] / std;

       // Standardize y for the fit and store it into the y_d array and the residual_y_d array
       for (int observation = 0; observation < num_observations; observation++) {
           float standardized_value = y_d[observation_thread_stride + observation] / std;
           y_d[observation_thread_stride + observation] = standardized_value;
           residual_y_d[observation_thread_stride + observation] = standardized_value;
       }
   }
}

} 


                         
// Define the GPU kernel that performs least squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
__global__ void model_fit(float * B_d, double * B_thread_stride_d, float * model_fit_flag_d, float * X_matrix_d, double * X_matrix_thread_stride_d, double * observation_thread_stride_d, float * residual_y_d, float * y_std_d, float * standardized_lambda_values_d, double * num_observations_d, double * num_predictors_d, float * alpha_values_d, float * tolerance_values_d, float * max_iterations_values_d, float * intercept_flag_d, int transformation_flag, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last beam block has less threads
if (block_thread_ind < num_threads_per_block_2) { 
   // Obtain the flag that determines whether to perform a model fit or not
   int model_fit_flag = (int)model_fit_flag_d[fit_ind];

   // This if statement is to ensure that a model fit is performed only if the model fit flag is 1
   if (model_fit_flag == 1) {
      // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
      int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];

      // Obtain the thread stride that is used to obtain the correct model matrix for the fit
      int X_thread_stride = (int)X_matrix_thread_stride_d[fit_ind];

      // Obtain the thread stride that is used to obtain the correct set of observations for the fit
      int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

      // Obtain the alpha value for the fit
      float alpha = alpha_values_d[fit_ind];

      // Obtain the standardized lambda value for the fit
      float lambda = standardized_lambda_values_d[fit_ind];

      // Obtain the tolerance value for the fit
      float tolerance = tolerance_values_d[fit_ind];

      // Obtain the max iterations value for the fit
      int max_iterations = (int)max_iterations_values_d[fit_ind];

      // Obtain the number of observations for the fit
      int num_observations = (int)num_observations_d[fit_ind];

      // Obtain the number of predictors for the fit
      int num_predictors = (int)num_predictors_d[fit_ind];

      // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
      int intercept_flag = (int)intercept_flag_d[fit_ind];

      // Declare and initialize the variable that stores the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values for one iteration of cyclic coordinate descent
      float global_max_change = 1E12;

      // Declare and initialize the variable that counts how many iterations of cyclic coordinate descent have been performed
      int iteration_count = 0;

      // Perform cyclic coordinate descent until either the maximum number of iterations is reached or the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values becomes less than the tolerance
      while (global_max_change >= tolerance && iteration_count < max_iterations) {
            // Declare and initialize the variable that stores the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values for one iteration of cyclic coordinate descent
            float max_change = 0.0f;

            // Declare and initialize the variable that stores the weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values that are due to the current predictor coefficient being updated using cyclic coordinate descent
            float change = 0.0f;
         
            // Cycle through all of the predictors for one iteration of cyclic coordinate descent
            for (int j = 0; j < num_predictors; j++) {
                // Obtain the predictor coefficient value for the current predictor
                float B_j = B_d[predictor_thread_stride + j];

                // Store the predictor coefficent value before it's updated
                float previous_B_j = B_j;
       
                // Declare and initialize the variable that stores the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                float p_j = 0.0f;

                // Calculate the residual values leaving the current predictor out (the predictor coefficients are initialized to zero, so the residual values are going to initially be y)
                // This if-else statement accounts for the fact that the contribution of the current predictor only needs to be removed from the residual values if the predictor coefficient is not zero
                // This is due to the fact that if the predictor coefficient is already zero, then the predictor contribution to the residual is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];
 
                       // Remove the contribution of the current predictor from the current residual value
                       float residual_y_value = residual_y_d[observation_thread_stride + observation_row] + (X_value * B_j);

                       // Store the updated residual value back into the residual_y_d array
                       residual_y_d[observation_thread_stride + observation_row] = residual_y_value;
      
                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out 
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);
                   }
                } else {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];

                       // Obtain the residual value (this is essentially the residual value leaving the current predictor out because the predictor coefficient value is zero) 
                       float residual_y_value = residual_y_d[observation_thread_stride + observation_row];

                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);        
                   }
                } 

                // Divide the computed correlation by the total number of observations in y (also the total number of observations in one predictor column)
                p_j = (1.0f / (float)num_observations) * p_j;

                // Apply the soft-thresholding function that is associated with the L1 regularization component of elastic-net regularization 
                float gamma = lambda * alpha;
                if (p_j > 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j - gamma;
                } else if (p_j < 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j + gamma;
                } else {
                   B_j = 0.0f;
                }

                // Declare and initialize the mean of the square of the predictor column
                float mean_squared_predictor_value = 0.0f;

                // Obtain the mean of the square of the predictor column
                if (transformation_flag == 1 || transformation_flag == 3) {
                   mean_squared_predictor_value = 1.0f;
                } else if (transformation_flag == 2 || transformation_flag == 4) {
                   mean_squared_predictor_value = 1.0f / (float)num_observations;
                }

                // This if-else statemet accounts for the fact that regularization is not applied to the intercept term if one is included
                if (intercept_flag == 1 && j == 0) {
                   // Use the computed correlation value as the updated predictor coefficient 
                   B_j = p_j;
                } else {
                   // Calculate the updated predictor coefficient value by applying the component of elastic-net regularization that is associated with L2 regularization 
                   B_j = B_j / (mean_squared_predictor_value + (lambda * (1.0f - alpha)));
                }

                // Store the updated predictor coefficient value into the B_d array
                B_d[predictor_thread_stride + j] = B_j;

                // Update the residual values to include the contribution of the current predictor using the updated predictor coefficient value 
                // If the updated predictor coefficient value is 0, then it's contribution to the residual values is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Store the updated residual back into the residual_y_d array
                       residual_y_d[observation_thread_stride + observation_row] = residual_y_d[observation_thread_stride + observation_row] - (X_matrix_d[X_thread_stride + (j * num_observations) + observation_row] * B_j);
                   }
                }
      
                // Compute the weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values (this is used for the tolerance convergence criterion)
                change = (previous_B_j - B_j) * (previous_B_j - B_j);
                if (transformation_flag == 2 || transformation_flag == 4) {
                    if (intercept_flag == 1 && j > 0) {
                       change = (1.0f / (float)num_observations) * change;
                    } else if (intercept_flag == 0) {
                       change = (1.0f / (float)num_observations) * change;   
                    }
                }
                if (change > max_change) {
                   max_change = change;
                }
            }
   
            // Update the global_max_change variable
            global_max_change = max_change;
        
            // Update the iteration count variable
            iteration_count = iteration_count + 1;
      }
 

      // Account for the fact that the y in the model fit was divided by its standard deviation
      float std_y = y_std_d[fit_ind];
      for (int j = 0; j < num_predictors; j++) {
          B_d[predictor_thread_stride + j] = B_d[predictor_thread_stride + j] * std_y;
      }
   }
}

}




// Define the GPU kernel that performs least squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the model matrices to the data
__global__ void model_fit_shared_memory(float * B_d, double * B_thread_stride_d, float * model_fit_flag_d, float * X_matrix_d, double * X_matrix_thread_stride_d, double * observation_thread_stride_d, float * residual_y_d, float * y_std_d, float * standardized_lambda_values_d, double * num_observations_d, double * num_predictors_d, float * alpha_values_d, float * tolerance_values_d, float * max_iterations_values_d, float * intercept_flag_d, int transformation_flag, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Define the shared memory array that stores the residual values of the model fits within one block (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last beam block has less threads
if (block_thread_ind < num_threads_per_block_2) { 
   // Obtain the flag that determines whether to perform a model fit or not
   int model_fit_flag = (int)model_fit_flag_d[fit_ind];

   // This if statement is to ensure that a model fit is performed only if the model fit flag is 1
   if (model_fit_flag == 1) {
      // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
      int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];

      // Obtain the thread stride that is used to obtain the correct model matrix for the fit
      int X_thread_stride = (int)X_matrix_thread_stride_d[fit_ind];

      // Obtain the thread stride that is used to obtain the correct set of observations for the fit
      int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

      // Obtain the alpha value for the fit
      float alpha = alpha_values_d[fit_ind];

      // Obtain the standardized lambda value for the fit
      float lambda = standardized_lambda_values_d[fit_ind];

      // Obtain the tolerance value for the fit
      float tolerance = tolerance_values_d[fit_ind];

      // Obtain the max iterations value for the fit
      int max_iterations = (int)max_iterations_values_d[fit_ind];

      // Obtain the number of observations for the fit
      int num_observations = (int)num_observations_d[fit_ind];

      // Obtain the number of predictors for the fit
      int num_predictors = (int)num_predictors_d[fit_ind];

      // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
      int intercept_flag = (int)intercept_flag_d[fit_ind];

      // Declare and initialize the variable that stores the maximum predictor coefficient change
      float global_max_change = 1E12;

      // Declare and initialize the variable that counts how many iterations of cyclic coordinate descent have been performed
      int iteration_count = 0;

      // Store the residual values for the fit into the shared memory array
      for (int observation_row = 0; observation_row < num_observations; observation_row++) {
          int store_ind = (observation_row * num_threads_per_block) + block_thread_ind;
          sdata[store_ind] = residual_y_d[observation_thread_stride + observation_row];
      }

      // Perform cyclic coordinate descent until either the maximum number of iterations convergence criterion or the tolerance criterion is met 
      while (global_max_change >= tolerance && iteration_count < max_iterations) {
            // Declare and initialize the variable that stores the maximum predictor coefficient change for one iteration
            float max_change = 0.0f;

            // Declare and initialize the variable that stores the change between the current predictor coefficient value and its previous value
            float change = 0.0f;
         
            // Cycle through all of the predictors for one iteration of cyclic coordinate descent
            for (int j = 0; j < num_predictors; j++) {
                // Obtain the predictor coefficient value for the current predictor
                float B_j = B_d[predictor_thread_stride + j];

                // Store the predictor coefficent value before it's updated
                float previous_B_j = B_j;
       
                // Declare and initialize the variable that stores the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                float p_j = 0.0f;

                // Calculate the residual values leaving the current predictor out (the predictor coefficients are initialized to zero, so the residual values are going to initially be y)
                // This if-else statement accounts for the fact that the contribution of the current predictor only needs to be removed from the residual values if the predictor coefficient is not zero
                // This is due to the fact that if the predictor coefficient is already zero, then the predictor contribution to the residual is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];
 
                       // Remove the contribution of the current predictor from the current residual value
                       float residual_y_value = sdata[(observation_row * num_threads_per_block) + block_thread_ind] + (X_value * B_j);

                       // Store the updated residual value back into the shared memory array
                       sdata[(observation_row * num_threads_per_block) + block_thread_ind] = residual_y_value;
      
                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out 
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);
                   }
                } else {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];

                       // Obtain the residual value (this is essentially the residual value leaving the current predictor out because the predictor coefficient value is zero) 
                       float residual_y_value = sdata[(observation_row * num_threads_per_block) + block_thread_ind];

                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);
                   }
                } 

                // Divide the computed correlation by the total number of observations in y (also the total number of observations in one predictor column)
                p_j = (1.0f / (float)num_observations) * p_j;

                // Apply the soft-thresholding function that is associated with the L1 regularization component of elastic-net regularization 
                float gamma = lambda * alpha;
                if (p_j > 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j - gamma;
                } else if (p_j < 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j + gamma;
                } else {
                   B_j = 0.0f;
                }

                // Declare and initialize the mean of the square of the predictor column
                float mean_squared_predictor_value = 0.0f;

                // Obtain the mean of the square of the predictor column
                if (transformation_flag == 1 || transformation_flag == 3) {
                   mean_squared_predictor_value = 1.0f;
                } else if (transformation_flag == 2 || transformation_flag == 4) {
                   mean_squared_predictor_value = 1.0f / (float)num_observations;
                }

                // This if-else statemet accounts for the fact that regularization is not applied to the intercept term if one is included
                if (intercept_flag == 1 && j == 0) {
                   // Use the computed correlation value as the updated predictor coefficient
                   B_j = p_j;
                } else {
                   // Calculate the updated predictor coefficient value by applying the component of elastic-net regularization that is associated with L2 regularization 
                   B_j = B_j / (mean_squared_predictor_value + (lambda * (1.0f - alpha)));
                }

                // Store the updated predictor coefficient value into the B_d array
                B_d[predictor_thread_stride + j] = B_j;

                // Update the residual values to include the contribution of the current predictor using the updated predictor coefficient value 
                // If the updated predictor coefficient value is 0, then it's contribution to the residual values is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Store the updated residual back into the shared memory array
                       sdata[(observation_row * num_threads_per_block) + block_thread_ind] = sdata[(observation_row * num_threads_per_block) + block_thread_ind] - (X_matrix_d[X_thread_stride + (j * num_observations) + observation_row] * B_j);
                   }
                }
      
                // Compute the weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values (this is used for the tolerance convergence criterion)
                change = (previous_B_j - B_j) * (previous_B_j - B_j);
                if (transformation_flag == 2 || transformation_flag == 4) {
                    if (intercept_flag == 1 && j > 0) {
                       change = (1.0f / (float)num_observations) * change;
                    } else if (intercept_flag == 0) {
                       change = (1.0f / (float)num_observations) * change;   
                    }
                }
                if (change > max_change) {
                   max_change = change;
                }
            }
   
            // Update the global_max_change variable
            global_max_change = max_change;
        
            // Update the iteration count variable
            iteration_count = iteration_count + 1;
      }
 

      // Account for the fact that the y in the model fit was divided by its standard deviation
      float std_y = y_std_d[fit_ind];
      for (int j = 0; j < num_predictors; j++) {
          B_d[predictor_thread_stride + j] = B_d[predictor_thread_stride + j] * std_y;
      }
   }
}

}




// Define the GPU kernel that performs predictor coefficient unnormalization
__global__ void predictor_coefficient_unnormalization(float * B_d, double * B_thread_stride_d, float * model_fit_flag_d, float * X_matrix_d, double * X_matrix_thread_stride_d, float * scaling_factors_d, double * num_predictors_d, float * intercept_flag_d, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) { 
   // Obtain the flag that determines whether a model fit was performed or not
   int model_fit_flag = (int)model_fit_flag_d[fit_ind];

   // This if statement is to ensure that the coefficients are unnormalized only if a model fit was performed
   if (model_fit_flag == 1) {
      // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
      int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];  

      // Obtain the number of predictors for the fit
      int num_predictors = (int)num_predictors_d[fit_ind];

      // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
      int intercept_flag = (int)intercept_flag_d[fit_ind];

      // Declare and initialize the variable that stores the number of the first predictor column to be standardized
      int start_ind = 0;

      // This if statement makes sure to not standardize the first predictor column if it corresponds to the intercept term
      if (intercept_flag == 1) {
         start_ind = 1;
      }

      // Reconstruct the signal value for the current observation by doing X_ROI * B_ROI
      for (int predictor_column = start_ind; predictor_column < num_predictors; predictor_column++) {
          B_d[predictor_thread_stride + predictor_column] = B_d[predictor_thread_stride + predictor_column] / scaling_factors_d[predictor_thread_stride + predictor_column];
      }
   }
}

}




// Define the GPU kernel that performs predictor coefficient unnormalization
__global__ void predictor_coefficient_unstandardization(float * B_d, double * B_thread_stride_d, float * model_fit_flag_d, float * X_matrix_d, double * X_matrix_thread_stride_d, float * scaling_factors_d, float * mean_X_matrix_d, double * num_predictors_d, float * intercept_flag_d, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) { 
   // Obtain the flag that determines whether a model fit was performed or not
   int model_fit_flag = (int)model_fit_flag_d[fit_ind];

   // This if statement is to ensure that the coefficients are unstandardized only if a model fit was performed
   if (model_fit_flag == 1) {
      // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
      int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];  

      // Obtain the number of predictors for the fit
      int num_predictors = (int)num_predictors_d[fit_ind];

      // Obtain the flag that determines whether the first predictor column is a column of ones for the intercept term or not
      int intercept_flag = (int)intercept_flag_d[fit_ind];

      // Declare and initialize the variable that stores the number of the first predictor column to be standardized
      int start_ind = 0;

      // This if statement makes sure to not standardize the first predictor column if it corresponds to the intercept term
      if (intercept_flag == 1) {
         start_ind = 1;
      }

      // Declare and initialize the variable that is used to adjust the intercept term if it is included
      float sum_value = 0.0f;

      // Reconstruct the signal value for the current observation by doing X_ROI * B_ROI
      for (int predictor_column = start_ind; predictor_column < num_predictors; predictor_column++) {
          float B_unstandardized = B_d[predictor_thread_stride + predictor_column] / scaling_factors_d[predictor_thread_stride + predictor_column];
          B_d[predictor_thread_stride + predictor_column] = B_unstandardized;
          sum_value = sum_value + (B_unstandardized * mean_X_matrix_d[predictor_thread_stride + predictor_column]);
      }

      // Adjust the intercept term if it is included
      if (intercept_flag == 1) {
          B_d[predictor_thread_stride] = B_d[predictor_thread_stride] - sum_value;
      }
   }
}

}
