function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
s1 = 0;
s2 = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    for i = 1:m
        t1 = (theta' * X(i,:)' - y(i)); % (h(x) - y)
        xi = X(:,2)(i);
        t2 = (theta' * X(i,:)' - y(i)) * xi; % (h(x) - y) * x
        s1 = s1 + t1;
        s2 = s2 + t2;
    end
    % fprintf('s1 = %f\n', s1);
    % fprintf('s2 = %f\n', s2);
    % simultaneously update for theta - the first formular in page 6 in ex1.pdf
    theta(1) = theta(1) - alpha * s1 / m;
    theta(2) = theta(2) - alpha * s2 / m;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    s1 = 0;
    s2 = 0;
    % fprintf('\nCost function  = %f at step %d', J_history(iter), iter);
end

end
