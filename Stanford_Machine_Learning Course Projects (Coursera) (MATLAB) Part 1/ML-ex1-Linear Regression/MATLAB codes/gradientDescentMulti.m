function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    % Instructions: Perform a single gradient step on the parameter vector theta. 

    % ====================== YOUR CODE HERE ======================


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

GD = zeros(m,size(X,2));
temp = zeros(size(X,2),1);

for iter = 1:num_iters


   		

	for j = 1:size(X,2),
		GD(:,j) = (X*theta.-y).*X(:,j);
	end;	
		
	for k = 1:size(X,2),
		temp(k) = theta(k)-alpha*(1/m)*sum(GD(:,k));
	end;	
	
	for l = 1:size(X,2),
		theta(l) = temp(l);
	end;	


    

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

	% ==========================================================