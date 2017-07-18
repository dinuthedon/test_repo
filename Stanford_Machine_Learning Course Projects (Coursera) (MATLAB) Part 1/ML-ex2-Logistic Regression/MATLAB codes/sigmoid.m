function g = sigmoid(z)

% Instructions: Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

% ====================== YOUR CODE HERE ======================

g = zeros(size(z));



for i = 1:size(z,1),
   for j = 1:size(z,2),
	g(i,j) = (1+exp(-z(i,j)))^-1;
   end;
end;



% =============================================================

end
