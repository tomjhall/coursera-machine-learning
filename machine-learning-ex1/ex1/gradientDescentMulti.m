function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
numFeatures = length(theta);
next_theta = theta;
sumOf = zeros(numFeatures,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %




    for feature = 1:numFeatures
	    sum = 0;
	    for sumIterator = 1 : m
		    h = X(sumIterator,:) * theta;
		    % h = theta(1) * X(sumIterator,1) + theta(2) * X(sumIterator,2);
		    % h = theta(1) + theta(2) * X(sumIterator,2);
		    sum += (h - y(sumIterator)) * X(sumIterator,feature);
	    end

       next_theta(feature) = theta(feature) - alpha / m * sum;
    end

    theta =  next_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
