function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
% C_vec = [0.01];
% sigma_vec = [0.01];

errors = zeros(length(C_vec), length(sigma_vec));

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

for c = 1:length(C_vec)
    for s = 1:length(sigma_vec)
        C = C_vec(c);
        sigma = sigma_vec(s);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        errors(c, s) = mean(double(predictions ~= yval));
    end;
end;

[~, c_idx] = min(min(errors, [], 2));
[~, sigma_idx] = min(min(errors, [], 1));
C = C_vec(c_idx);
sigma = sigma_vec(sigma_idx);
fprintf('Found C = %f | sigma = %f\n', C, sigma);

% Found C = 1.000000 | sigma = 0.100000
% =========================================================================

end
