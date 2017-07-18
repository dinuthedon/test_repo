function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.

% ====================== YOUR CODE HERE ======================

m = size(X, 1); % Number of training examples


error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i = 1:m,

	theta = trainLinearReg(X((1:i),:), y(1:i), lambda);

	h = X(1:i,:)*theta;

	error_train(i) = (sum((h-y(1:i)).^2))/(2*i);

	hval = Xval*theta;

	error_val(i) = (sum((hval-yval).^2))/(2*size(yval,1));

end;	

% =========================================================================

end
