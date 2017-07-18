function [X_norm, mu, sigma] = featureNormalize(X)

% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 


% ====================== YOUR CODE HERE ======================

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));


mu = mean(X);
sigma = std(X);

for i=1:size(X,1),
	for j=1:size(X,2),

		X_norm(i,j) = (X(i,j)-mu(j))/sigma(j);

	end;
end;

% ============================================================

end
