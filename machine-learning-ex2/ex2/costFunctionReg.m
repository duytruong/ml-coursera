function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
nvar = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


s = 0;
s_reg = 0;
for i = 1:m
    t = -y(i) * log(sigmoid(theta' * X(i,:)')) - (1 - y(i)) * log(1 - sigmoid(theta' * X(i,:)'));
    s = s + t;
end;

for j = 2:nvar
    reg = theta(j,:)^2;
    s_reg = s_reg + reg;
end;
J = ((lambda * s_reg) / (2 * m)) + (s / m);

for j = 1:nvar
    s = 0;
    reg = 0;
    for i = 1:m
        g = (sigmoid(theta' * X(i,:)') - y(i)) * X(i,j);
        s = s + g;
    end;
    if (j > 1)
        reg = lambda * theta(j,:) / m;
    endif
    grad(j,1) = s / m + reg;
end;



% =============================================================

end
