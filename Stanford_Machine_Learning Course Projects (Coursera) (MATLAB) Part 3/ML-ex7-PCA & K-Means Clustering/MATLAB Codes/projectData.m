function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors

% Instructions: Compute the projection of the data using only the top K eigenvectors in U (first K columns). 

% ====================== YOUR CODE HERE ======================


Z = zeros(size(X, 1), K);

for k = 1:K,

	for i = 1:size(X,1),

		x = X(i,:)';

		
		Z(i,k) = x'*U(:,k);
	end;
end;


% =============================================================

end
