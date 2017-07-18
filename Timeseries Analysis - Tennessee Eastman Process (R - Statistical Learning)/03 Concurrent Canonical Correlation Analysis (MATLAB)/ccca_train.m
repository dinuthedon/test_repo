%=================================================================
% Function for training CCCA model based on scaled input X0 and Y0
%=================================================================


function [R, T,  Q, P, Pc, Rc, Qc, Px, Qy,lamda_c,lamda_x, lamda_y, Ac, Ax, Ay, mm, pp,PHI_y, ...
    Tc2_lim,Tx2_lim,Qx_lim,Ty2_lim, Qy_lim, phi_y_lim] = ccca_train(X0, Y0, a)

[n, m] = size(X0);

%% Step 1: CCA

[R,T,Q,P] = cca(X0, Y0, a);
lamda = T'*T;

%% Step 2: Getting predictable part of the output data

Yh = X0*R*Q';
[U0, D0, V0] = svd(Yh);

if size(D0,2) == 1
    Sc = D0(1);     
else
    Sc = diag(D0); 
    Sc = Sc(diag(D0)>0);
end

com = 0;
Ac = 0;

for i = 1:length(Sc)
    com = com + Sc(i);
    Ac = Ac+1;
    if com > 0.95*sum(Sc)
        break
    end
end

Sc = Sc(1:Ac);
lamda_c = 1/(n-1)*diag(Sc.^2);

Qc = V0(:,1:Ac);
Tc = Yh*Qc;

%% Step 3: Getting unpredictable part of the output data

Yct = Y0 - Tc*Qc';
[U1,D1,V1]=svd(Yct);

Ay = pc_number(Yct);

Qy = V1(:,1:Ay);
Ty = Yct*Qy;     		%Output-principal scores
Yt = Yct - Ty*Qy';

if size(D1,2) == 1
    Sy = D1(1);
else
    Sy = diag(D1);
end

pp = length(Sy);
lamda_y = 1/(n-1)*diag(Sy(1:Ay).^2);

%% Step 4: Getting Output-Irrelevant part of the input data

Pc = (inv(Tc'*Tc)*Tc'*X0)';
Rc = R*Q'*Qc;

Xct = X0 - Tc*pinv(Rc);

[U2,D2,V2] = svd(Xct);

Ax = pc_number(Xct);

Px = V2(:,1:Ax);
Tx = Xct*Px; 			% Input-principal scores

if size(D2,2) == 1
    Sx = D2(1);
else
    Sx = diag(D2);
end

mm = length(Sx);
lamda_x = 1/(n-1)*diag(Sx(1:Ax).^2);

%% CCCA control limit for fault-detection

alpha = 0.01;
level = 1-alpha; 

gx = 1/(n-1)*sum(Sx(Ax+1:mm).^4)/sum(Sx(Ax+1:mm).^2);
hx = (sum(Sx(Ax+1:mm).^2))^2/sum(Sx(Ax+1:mm).^4);
gy = 1/(n-1)*sum(Sy(Ay+1:pp).^4)/sum(Sy(Ay+1:pp).^2);
hy = (sum(Sy(Ay+1:pp).^2))^2/sum(Sy(Ay+1:pp).^4);
Ac = 3;

Tc2_lim = chi2inv(level, Ac);   	%Output-relevant fault based on x

Tx2_lim = chi2inv(level, Ax);   	%Output-irrelevant but input-relevant based on x

Qx_lim = gx*chi2inv(level, hx); 	% Potentially Output-relevant based on x

Ty2_lim = chi2inv(level, Ay);    % Output relevant

Qy_lim = gy*chi2inv(level, hy);	% Output relevant

%combined index

SS_y = 1/(n-1)*(Yct'*Yct); 		% sample covariance

PHI_y = Qy*inv(lamda_y)*Qy'/Ty2_lim + (eye(size(Qy*Qy'))-Qy*Qy')/Qy_lim;

g_phi = trace((SS_y*PHI_y)^2)/trace(SS_y*PHI_y);

h_phi = (trace(SS_y*PHI_y))^2/trace((SS_y*PHI_y)^2);

phi_y_lim = g_phi*chi2inv(level, h_phi); % control limit of combined index

