function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% ====================== YOUR CODE HERE ======================

error_train = zeros(length(lambda_vec), 1);

error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec),

	lambda = lambda_vec(i);

	theta = trainLinearReg(X, y, lambda);

	h = X*theta;

	
	error_train(i) = (sum((h-y).^2))/(2*size(y,1));

	hval = Xval*theta;

	error_val(i) = (sum((hval-yval).^2))/(2*size(yval,1));

% =========================================================================

end
