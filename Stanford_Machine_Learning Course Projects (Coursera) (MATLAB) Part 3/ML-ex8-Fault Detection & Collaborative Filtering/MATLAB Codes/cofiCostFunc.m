function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function

% Instructions: Compute the cost function and gradient for collaborative filtering. 


% ====================== YOUR CODE HERE ======================

% Unfold the U and W matrices from params

X = reshape(params(1:num_movies*num_features), num_movies, num_features);

Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

        

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
% sum = 0;

% for i = 1:num_movies,

	%for j = 1:num_users,

		%if (R(i,j) == 1),

			%sum = sum + (Theta(j,:)*(X(i,:))'- Y(i,j))^2;
		%end;
	%end;
%end;

reg = lambda / 2 * (sum(sum(Theta .^2 )) + sum(sum(X .^2 )));

sum = sum(sum(R.*((X*Theta'-Y).^2)));


J = sum/2 + reg;


X_grad = R.*(X*Theta' - Y)*Theta + lambda*X;

Theta_grad = (R.*(X*Theta' - Y))'*X + lambda*Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
