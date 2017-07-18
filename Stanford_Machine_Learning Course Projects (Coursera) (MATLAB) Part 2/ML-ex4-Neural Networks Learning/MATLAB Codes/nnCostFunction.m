function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively
%
% Part 3: Implement regularization with the cost function and gradients.

% ====================== YOUR CODE HERE ======================

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];
         
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

temp1 = Theta1_grad;
temp2 = Theta2_grad;

A = temp1;
B = temp2;


yv = zeros(m,num_labels);
for i = 1:m,
	yv(i,y(i)) = 1;
end;

% Leading up to h

A1 = X;

Z2 = A1*Theta1';

A2 = sigmoid(Z2);

A2 = [ones(m,1) A2];

Z3 = A2*Theta2';

A3 = sigmoid(Z3);

h = A3;

V = 0;


for i = 1:m,

	for k = 1:num_labels,

	V = V+(-(yv(i,k).*log(h(i,k)))-((1-yv(i,k)).*log(1-h(i,k))));

	end;
	
end;

T1 = zeros(size(Theta1));

for  i = 2:size(Theta1,2),

	for j = 1:size(Theta1,1),
	
		T1(j,i) = Theta1(j,i).^2;

	end;
end;

T2 = zeros(size(Theta2));

for  i = 2:size(Theta2,2),

	for j = 1:size(Theta2,1),
	
		T2(j,i) = Theta2(j,i).^2;

	end;
end;



J = ((1/m)*V)+(lambda/(2*m)*(sum(T1(:))+sum(T2(:))));



% Gradient Delta Matrix Generation

for t = 1:m,

	a1 = X(t,:);					% 1x401

	z2 = a1*Theta1';					% 1x25

	a2 = sigmoid(z2);				% 1x25

	a2 = [1 a2];					% 1x26

	z3 = a2*Theta2';					% 1x10

	a3 = sigmoid(z3);				% 1x10

	
	d3 = a3.-yv(t,:);				% 1x10

	z2n = [1 z2];					% 1x26

	d2 = (d3*Theta2).*(sigmoidGradient(z2n)); % 1x26

	temp2 = temp2 + d3'*a2;			% 10x26

	d2n = d2(2:hidden_layer_size+1);		% 1x25

	temp1 = temp1 + d2n'*a1;			% 25x401

end;


for i = 1:size(Theta1,1),
	for j = 2:size(Theta1,2),

		A(i,j) = Theta1(i,j);
	end;
end;

for i = 1:size(Theta2,1),
	for j = 2:size(Theta2,2),

		B(i,j) = Theta2(i,j);
	end;
end;



Theta1_grad = (temp1+lambda*A)/m;
Theta2_grad = (temp2+lambda*B)/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
