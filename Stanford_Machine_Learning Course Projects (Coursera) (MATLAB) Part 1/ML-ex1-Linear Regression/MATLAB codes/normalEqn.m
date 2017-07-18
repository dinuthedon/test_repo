function [theta] = normalEqn(X, y)

% Instructions: Complete the code to compute the closed form solution to linear regression and put the result in theta.

% ====================== YOUR CODE HERE ======================

theta = zeros(size(X, 2), 1);

theta = pinv(X'*X)*X'*y;


end
