%=================================================================
% Zero-mean Scaling of data:
%=================================================================


function data = autos_test(data,m_train,v_train)

[mm,nn]=size(data);				% data: sample*variable
for i=1:nn
    M(:,i)=ones(mm,1)*m_train(i);    	% mean 
    data(:,i)=(data(:,i)-M(:,i))/v_train(i); % auto scaled

end
