function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta

%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

J = 0;



V=X*theta.-y;
Vsq=V.^2;

J=1/(2*m)*sum(Vsq);

% =========================================================================

end
