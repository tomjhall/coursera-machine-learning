function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %=============

    sum = 0;
    for sumIterator = 1 : m
	    h = X(sumIterator,:) * theta;
	    % h = theta(1) * X(sumIterator,1) + theta(2) * X(sumIterator,2);
	    % h = theta(1) + theta(2) * X(sumIterator,2);
	    sum += (h - y(sumIterator)) * X(sumIterator,1);
    end

    thetaZero = theta(1) - alpha / m * sum;

    sum = 0;
    for sumIterator = 1 : m
	    h = X(sumIterator,:) * theta;
	    % h = theta(1) * X(sumIterator,1) + theta(2) * X(sumIterator,2);
	    % h = theta(1) + theta(2) * X(sumIterator,2);
	    sum += (h - y(sumIterator)) * X(sumIterator,2);
    end

    thetaOne = theta(2) - alpha / m * sum;

    theta(1) = thetaZero;
    theta(2) = thetaOne;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
