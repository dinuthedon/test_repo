function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data

% Instructions: Compute the approximation of the data by projecting back onto the original space using the top K eigenvectors in U.

% ====================== YOUR CODE HERE ======================

X_rec = zeros(size(Z, 1), size(U, 1));

for i = 1:size(Z,1),

	for j = 1:size(U,1),
	
		v = Z(i,:)';

		X_rec(i,j) = v' * U(j, 1:K)';
	end;
end;


% =============================================================

end
