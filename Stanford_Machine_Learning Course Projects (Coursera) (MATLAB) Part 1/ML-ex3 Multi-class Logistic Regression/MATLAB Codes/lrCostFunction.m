function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ====================== YOUR CODE HERE ======================

m = length(y); % number of training examples

J = 0;

grad = zeros(size(theta));

h = sigmoid(X*theta);

V = -y.*log(h)-(1-y).*log(1-h);

T = zeros(size(theta));

for j = 2:size(theta),
	
	T(j) = theta(j).^2;
end;

t = lambda/(2*m)*sum(T);

J = (1/m)*sum(V)+t;

W = zeros(m,size(theta));


for j = 1:size(theta),

		W(:,j) = (h.-y).*X(:,j);

		theta(1) = 0;

		grad(j) = ((1/m)*sum(W(:,j)))+(lambda/m)*theta(j);
end;

% =============================================================

grad = grad(:);

end
