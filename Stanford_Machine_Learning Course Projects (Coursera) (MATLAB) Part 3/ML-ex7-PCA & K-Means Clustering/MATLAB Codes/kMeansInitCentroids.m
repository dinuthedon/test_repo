function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X

% Instructions: You should set centroids to randomly chosen examples from  the dataset X

% ====================== YOUR CODE HERE ======================

centroids = zeros(K, size(X, 2));

randidx = randperm(size(X,1)); % Randomly reorder indices of X

centroids = X(randidx(1:K),:); % Take first K examples as centroids

% =============================================================

end

