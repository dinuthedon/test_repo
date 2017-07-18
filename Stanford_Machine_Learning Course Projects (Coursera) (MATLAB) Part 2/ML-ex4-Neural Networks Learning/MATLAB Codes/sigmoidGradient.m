function g = sigmoidGradient(z)

% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

% ====================== YOUR CODE HERE ======================

g = zeros(size(z));

g = sigmoid(z).*(1-sigmoid(z));

% =============================================================

end
