function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Instructions: Complete the following code to make predictions using your learned neural network. 

% ====================== YOUR CODE HERE ======================

m = size(X, 1);

num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X]; % Add ones to the X data matrix

A1 = sigmoid(X*Theta1');

A1 = [ones(m,1) A1]; % Add ones to the A1 matrix

h = sigmoid(A1*Theta2');

for i = 1:m,
	
	[a, ia] = max(h(i,:));
	
	p(i) = ia;
	
end;

% =========================================================================


end
