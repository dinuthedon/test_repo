function p = predictOneVsAll(all_theta, X)

%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).

m = size(X, 1);

num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X]; % Add ones to the X data matrix


h = sigmoid(X*all_theta');

for i = 1:m,
	
	[a, ia] = max(h(i,:));
	
	p(i) = ia;
	
end;

% =========================================================================


end
