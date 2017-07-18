function [J, grad] = costFunction(theta, X, y)

% Instructions: Compute the cost of a particular choice of theta.

% ====================== YOUR CODE HERE ======================

m = length(y); % number of training examples

J = 0;

grad = zeros(size(theta));

h = sigmoid(X*theta);

V = -y.*log(h)-(1-y).*log(1-h);

J = (1/m)*sum(V);

for j = 1:size(theta),

	W(:,j) = (h.-y).*X(:,j);

	grad(j) = (1/m)*sum(W(:,j));
end;


% =============================================================

end
