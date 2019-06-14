function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if (typeinfo(z) == 'matrix')
    for i = 1:size(z, 1) % i: row index
        for j = 1:size(z,2) % j: column index
            sig_z = 1 / (1 + e^-z(i,j));
            g(i,j) = sig_z;
        end;
    end;
else
    sig_z = 1 / (1 + e^-z);
    g = sig_z;
endif
% =============================================================

end
