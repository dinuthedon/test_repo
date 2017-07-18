function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% Instructions: Compute the cost and gradient of regularized linear regression for a particular choice of theta.

% ====================== YOUR CODE HERE ======================

m = length(y); % number of training examples

J = 0;

grad = zeros(size(theta));

h = X*theta;


J1 = (h-y).^2;

J2 = theta.^2;

J = (sum(J1) + lambda*sum(J2(2:size(theta,1))))/(2*m);


for j = 1:size(theta),

		W(:,j) = (h.-y).*X(:,j);

		theta(1) = 0;

		grad(j) = ((1/m)*sum(W(:,j)))+(lambda/m)*theta(j);
end;

% =========================================================================

grad = grad(:);

end
