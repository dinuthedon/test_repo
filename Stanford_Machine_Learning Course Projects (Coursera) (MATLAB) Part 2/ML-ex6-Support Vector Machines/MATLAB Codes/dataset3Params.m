function [C, sigma] = dataset3Params(X, y, Xval, yval)

%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
C = 1;

sigma = 0.3;

Cval = [0.01 0.03 0.1 0.3 1 3 10 30];

sigval = [0.01 0.03 0.1 0.3 1 3 10 30];

error = zeros(size(Cval,2),size(sigval,2));

for i = 1:size(Cval,2),

	for j = 1:size(sigval,2),

		model= svmTrain(X, y, Cval(i), @(x1, x2) gaussianKernel(x1, x2, sigval(j)));

		predictions = svmPredict(model, Xval);

		error(i,j) = mean(double(predictions ~= yval));

	end;

end;

[r,c] = find(error==min(min(error)));

C = Cval(r);

sigma = sigval(c);

% =========================================================================

end
