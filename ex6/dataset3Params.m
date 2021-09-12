function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Variables
C_train = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_train = [0.01;0.03;0.1;0.3;1;3;10;30];
m = size(C_train, 1);
n = size(sigma_train, 1);
error_val = ones(m*n, 3);

% Ensure that x1 and x2 are column vectors
x1 = X(:,1); 
x2 = X(:,2);

for i=1:m
    for j=1:n
        error_val(i*j, 1) = C_train(i);
        error_val(i*j, 2) = sigma_train(j);
        model = svmTrain(X, y, C_train(i), @(x1, x2) gaussianKernel(x1, x2, sigma_train(j)));
        predictions = svmPredict(model, Xval);
        error_val(i*j, 3) = mean(double(predictions ~= yval));
    end
end

% find minimum of prediction error
error_val_vec = error_val(:, 3);
error_val_min = min(error_val(:, 3));
find_error_val_min = error_val(find(error_val_vec == error_val_min), :);

C = find_error_val_min(1);
sigma = find_error_val_min(2);

% =========================================================================

end
