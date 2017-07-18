function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables

% ====================== YOUR CODE HERE ======================

%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the parameter for linear regression to fit the data points in X and y


m = length(y); % number of training examples


J = 0;


	V=X*theta.-y;
	
	J = 1/(2*m)*(V'*V);



% =========================================================================

end
