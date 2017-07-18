%=================================================================
% Function PLS using NIPALS algorithm:
%=================================================================


function [t,p,q,w] = pls(X0,Y0,a)

% a is the number of PLS factors

[n,m] = size(X0);
np = size(Y0, 2);

X = zeros(n,m,a);
Y = zeros(n,np,a);
X(:,:,1) = X0;
Y(:,:,1) = Y0;

for i = 1:a
    u(:,i) = Y(:,1,i);
    itererr = 1;

    while norm(itererr)>0.000001
        w(:,i)=X(:,:,i)'*u(:,i)/norm(X(:,:,i)'*u(:,i));
        t(:,i)=X(:,:,i)*w(:,i);
        q(:,i)=Y(:,:,i)'*t(:,i)/(t(:,i)'*t(:,i));
        u(:,i)=Y(:,:,i)*q(:,i);
        itererr=t(:,i)-X(:,:,i)*X(:,:,i)'*u(:,i)/norm(X(:,:,i)'*u(:,i));
    end

    p(:,i)=X(:,:,i)'*t(:,i)/(t(:,i)'*t(:,i));
    X(:,:,i+1)=X(:,:,i)-t(:,i)*p(:,i)';
    Y(:,:,i+1)=Y(:,:,i)-t(:,i)*q(:,i)';

end