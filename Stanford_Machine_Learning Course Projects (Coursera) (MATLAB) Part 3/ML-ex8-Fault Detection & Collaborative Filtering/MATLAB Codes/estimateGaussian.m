function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X

% Instructions: Compute the mean of the data and the variances

% ====================== YOUR CODE HERE ======================

[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

mu = (sum(X)/m)';

sigma2 = (sum((X-mu').^2)/m)';

% =============================================================


end
