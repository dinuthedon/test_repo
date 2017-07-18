function sim = gaussianKernel(x1, x2, sigma)

% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
% ====================== YOUR CODE HERE ======================

x1 = x1(:); x2 = x2(:);

sim = 0;

sim = exp(-sum((x1-x2).^2)/(2*sigma^2));

% =============================================================
    
end
