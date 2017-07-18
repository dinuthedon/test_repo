function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example

% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. 

% ====================== YOUR CODE HERE ======================

K = size(centroids, 1);

idx = zeros(size(X,1), 1);

Cv = zeros(size(X,1), size(centroids,1));

for i = 1:size(X,1),

	for j = 1:size(centroids,1),

		Z = (X(i,:)-centroids(j,:));

		Cv(i,j) = norm(Z);
	end;

	[a, id] = min(Cv(i,:));

	idx(i) = id;
end;

% =============================================================

end

