%=================================================================
% Performing CCCA on Test dataset based on a trained CCCA model:
% This function returns the following:
%	1. Tc^2 index of the Canonical Correlation Subspace
%	2. Tx^2 index of the Input Principal Subspace
%	3. Qx index of the Input Residual Subspace
%	4. Ty^2 index of the Output Prinipal Subspace
%	5. Qy index of the Output Residual Subspace
%	6. Phi_y index
%=================================================================

function [Tc2_index,Tx2_index,Qx_index,Ty2_index,Qy_index,phi_y_index]...
    = ccca_test(X_test, Y_test,R,T, Q, P, Pc, Rc, Qc, Px, Qy,lamda_c,lamda_x, lamda_y, Ac, Ax, Ay, mm, pp,PHI_y,...
    Tc2_lim,Tx2_lim,Qx_lim,Ty2_lim, Qy_lim, phi_y_lim)

n = size(X_test,1);

Tc2_m = R*Q'*Qc*inv(lamda_c)*Qc'*Q*R';
% Tc2_m = Rc*inv(lamda_c)*Rc';

Tx2_m = Px*inv(lamda_x)*Px';
Qx_m = eye(size(Px*Px'))-Px*Px';

Ty2_m = Qy*inv(lamda_y)*Qy';
Qy_m = eye(size(Qy*Qy'))-Qy*Qy';

count = 0;
countQ = 0;

for i = 1:n

    Tc2_index(i) = X_test(i,:)*Tc2_m*X_test(i,:)';
    if Tc2_index(i) > Tc2_lim
        count = count + 1;
    end

    tc = Qc'*Q*R'*X_test(i,:)';

    xct = X_test(i,:)'-(pinv(Rc))'*tc;

    yct(:,i) = Y_test(i,:)'-Qc*tc;

    Tx2_index(i) = xct'*Tx2_m*xct;
    Qx_index(i) = xct'*Qx_m*xct;

    if Qx_index(i) > Qx_lim
        countQ = countQ + 1;
    end
    
    Ty2_index(i)= yct(:,i)'*Ty2_m*yct(:,i);
    
    Qy_index(i) = yct(:,i)'*Qy_m*yct(:,i); 
    phi_y_index(i) = yct(:,i)'*PHI_y*yct(:,i);
end

count / n;
countQ / n;