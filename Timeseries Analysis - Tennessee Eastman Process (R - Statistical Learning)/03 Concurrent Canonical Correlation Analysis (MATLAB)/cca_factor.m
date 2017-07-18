%=================================================================
% Function for choosing # of CCA factors by cross-validation:
% Functions called for execution: cca_mse, cca_predict, cca
%=================================================================


function [a,mse_min,mse] = cca_factor(X0,Y0)

m = size(X0,2);

for i = 1:m
    fun = @(XTRAIN,YTRAIN,XTEST,YTEST)(cca_mse(XTRAIN,YTRAIN,XTEST,YTEST,i));
    vals = crossval(fun, X0,Y0);
    mse(i) = mean(vals);
end

[mse_min, a] = min(mse);


