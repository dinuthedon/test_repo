function p = predict(theta, X)

% Instructions: Complete the following code to make predictions using your learned logistic regression parameters. 

% ====================== YOUR CODE HERE ======================

m = size(X, 1); % Number of training examples

p = zeros(m, 1);


h = sigmoid(X*theta);

for i = 1:m,

	if h(i)>=0.5,
		p(i) =1;
	else,
		p(i) =0;
	end;
% =========================================================================


end
