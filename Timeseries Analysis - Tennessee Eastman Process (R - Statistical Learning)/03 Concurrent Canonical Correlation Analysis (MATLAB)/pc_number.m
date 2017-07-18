%=================================================================
% Function for calculating the # of Prinipal Components:
%  (# of PC's that capture 95% variance of the data)
%=================================================================

function pcnumber = pc_number(X)

[~, D, ~] = svd(X);

if size(D,2)== 1
    pcnumber = 1;
else
    S = diag(D);
    S = S(diag(D)>0);
    i = 0;
    var = 0;
    while var <.95*sum(S.^2)
        i = i+1;
        var = var+S(i)^2;
    end
    pcnumber = i;
end
