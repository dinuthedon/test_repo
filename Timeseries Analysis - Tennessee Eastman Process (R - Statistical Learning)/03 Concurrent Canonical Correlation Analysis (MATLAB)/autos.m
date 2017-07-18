%=================================================================
% Zero-mean Scaling of data:
%=================================================================

function [data, m, v] = autos(data)

[mm,nn]=size(data);				% data: sample*variable

m=mean(data);
v=std(data);

for i=1:nn

    M(:,i)=ones(mm,1)*m(i);			% mean 
    data(:,i)=(data(:,i)-M(:,i))/v(i); % auto scaled

end

