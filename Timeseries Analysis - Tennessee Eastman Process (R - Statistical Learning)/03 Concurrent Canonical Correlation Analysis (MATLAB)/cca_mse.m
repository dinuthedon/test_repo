%=================================================================
% Function for calculating Mean Squared Error of CCA prediction:
%=================================================================


function mse = cca_mse(X0,Y0,X_test,Y_test,a)

Y_predict = cca_predict(X0,Y0,X_test,a);

Y_error = Y_test - Y_predict;

mse = sum(sum(Y_error.^2)) /size(Y_predict,1);
