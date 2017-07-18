%=================================================================
% Function for CCA prediction:
%=================================================================

function Y_predict = cca_predict(X0,Y0,X_test,a)

[R, C, T, U] = cca(X0,Y0,a);

P = X0'*T;
Q = Y0'*T;

Y_predict = X_test*R*Q';