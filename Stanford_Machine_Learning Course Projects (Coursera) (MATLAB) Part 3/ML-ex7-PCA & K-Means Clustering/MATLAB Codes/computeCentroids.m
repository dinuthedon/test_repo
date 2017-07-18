function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.

% Instructions: Go over every centroid and compute mean of all points that belong to it. 

% ====================== YOUR CODE HERE ======================

[m n] = size(X);

centroids = zeros(K, n);

for i = 1:K,

	c = 0;

	sum = zeros(1, n);

	for j = 1:m,


		if (idx(j) == i),
		
			c = c+1;
	
			sum = sum + X(j,:);
		end;
	end;

	centroids(i,:) = (1/c)*sum;
end;

% =============================================================


end

