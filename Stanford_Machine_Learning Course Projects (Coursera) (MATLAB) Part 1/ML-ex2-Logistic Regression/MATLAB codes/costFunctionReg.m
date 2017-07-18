function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Instructions: Compute the cost of a particular choice of theta with regularization

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

W(:,1) = (h.-y).*X(:,1);

grad(1) = (1/m)*sum(W(:,1));


for j = 2:size(theta),

		W(:,j) = (h.-y).*X(:,j);

		grad(j) = ((1/m)*sum(W(:,j)))+(lambda/m)*theta(j);
end;


% =============================================================

end
