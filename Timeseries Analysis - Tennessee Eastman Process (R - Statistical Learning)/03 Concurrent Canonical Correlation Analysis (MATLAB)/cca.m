%=================================================================
% Function to perform Canonical Correlation Analysis:
%	Eigenvalue Decompositon & Singualr Value Decomposition
%=================================================================

function [R, T, Q, P] = cca(X0, Y0, a)

% CCA (Eigenvalue Decomposition and SVD)

n = size(X0, 1);
[R1,C1,Sz,T1,U1] = canoncorr(X0,Y0);
R = R1/sqrt(n);
C = C1/sqrt(n);
T = T1/sqrt(n);
U = U1/sqrt(n);

R = R(:,1:a);
C = C(:,1:a);
T = T(:,1:a);
U = U(:,1:a);
Q = Y0'*T;
P = X0'*T;