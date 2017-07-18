function W = randInitializeWeights(L_in, L_out)

% Instructions: Initialize W randomly so that we break the symmetry while training the neural network.

% ====================== YOUR CODE HERE ======================

W = zeros(L_out, 1 + L_in);

epsilon_init = 0.12;

W = rand(L_out, 1+L_in)*2*epsilon_init-epsilon_init;

% =========================================================================

end
