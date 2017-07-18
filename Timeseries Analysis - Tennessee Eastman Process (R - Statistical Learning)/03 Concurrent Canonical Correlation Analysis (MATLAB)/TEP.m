%================================================================
% Function for executing CCCA on Tennessee Eastman Proceess:
%================================================================


clc; clear; close all;

%% load training data 

fid = fopen('Dataset/d00.dat','r');

train_data = fopen(fid,'%f',[500,52]);

fclose(fid);

x_train = zeros(size(train_data,1)/5,33);

x_train(:,1:22) = train_data(5:5:500,1:22);   % process measurements

x_train(:,23:33) = train_data(5:5:500,42:52); % manipulated variable

y_train = train_data(5:5:500,37:41);          % quality measurements

N_train = size(x_train,1);

%% zero-mean scaling of the training data

[x_train, mx_train, vx_train] = autos(x_train);
[y_train, my_train, vy_train] = autos(y_train);

%% CCCA training

a = 3;

[R, T, Q, P, Pc, Rc, Qc, Px, Qy,lamda_c,lamda_x, lamda_y, Ac, Ax, Ay, mm, pp, PHI_y, ...
    Tc2_lim,Tx2_lim,Qx_lim,Ty2_lim, Qy_lim, phi_y_lim] = ccca_train(x_train,y_train,a);


%% Load Test dataset

fid = fopen('Dataset/d04_te.dat','r');

test_data = fopen(fid,'%f',[52,inf])';

fclose(fid);

x_test = zeros(size(test_data,1)/5,33);

x_test(:,1:22) = test_data(5:5:960,1:22);   % process measurements

x_test(:,23:33) = test_data(5:5:960,42:52); % manipulated variable

y_test = test_data(5:5:960,37:41);          % quality measurements

%% zero-mean scaling of the test dataset

x_test = autos_test(x_test, mx_train,vx_train);
y_test = autos_test(y_test, my_train,vy_train);

%% CCCA testing (Obtain CCCA indices for the test data)

[Tc2_index_test,Tx2_index_test,Qx_index_test,Ty2_index_test,Qy_index_test, phi_y_index_test]...
    = ccca_test(x_test, y_test, R, T, Q, P, Pc, Rc, Qc, Px, Qy,lamda_c,lamda_x, lamda_y, Ac, Ax, Ay, mm, pp,PHI_y, ...
    Tc2_lim,Tx2_lim,Qx_lim,Ty2_lim, Qy_lim, phi_y_lim);

%% Obtain CCCA indices for training data

[Tc2_index,Tx2_index,Qx_index,Ty2_index,Qy_index,phi_y_index]...
    = ccca_test(x_train, y_train,  R, T, Q, P, Pc, Rc, Qc, Px, Qy,lamda_c,lamda_x, lamda_y, Ac, Ax, Ay, mm, pp,PHI_y, ...
    Tc2_lim,Tx2_lim,Qx_lim,Ty2_lim, Qy_lim, phi_y_lim);

%% plot Generation

n = size(x_test,1);

%% plot together

figure;

subplot(2,2,1);
plot([1:n],[Tc2_index_test], 'b', [1:n],Tc2_lim*ones(1,n),'r', 'LineWidth', 1.5);
title('T_c^2');

subplot(2,2,2);
plot([1:n],[Ty2_index_test], 'b', [1:n], Ty2_lim*ones(1,n),'r', 'LineWidth', 1.5);
title('T_y^2');

subplot(2,2,3);
plot([1:n],[Tx2_index_test], 'b', [1:n],Tx2_lim*ones(1,n),'r', 'LineWidth', 1.5);
title('T_x^2');

subplot(2,2,4);
plot([1:n],[Qx_index_test], 'b', [1:n], Qx_lim*ones(1,n),'r', 'LineWidth', 1.5);
title('Q_x');
