function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting outliers

    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.

    % ====================== YOUR CODE HERE ======================

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
	prediction = (pval<epsilon);

	tp = sum((prediction == 1)&(yval == 1));

	fp = sum((prediction == 1)&(yval == 0));

	fn = sum((prediction == 0)&(yval == 1));

	p = tp/(tp+fp);

	r = tp/(tp+fn);

	F1 = 2*p*r/(p+r);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
