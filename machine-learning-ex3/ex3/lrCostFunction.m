function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


% theta is a 401x1 vector (starting as zeros)
% X is a 5000x401 matrix (first column is 1s and the rest 400 are unrolled pixels as a row)
% y is a 5000x1 vector (where it's 1s where the training set is right, and 0 everywhere else - 500 will be 1 and 4500 will be 0)
% lambda is just a scalar - 0.1 for this...

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% 

hTheta = sigmoid(X * theta);

J = sum(log(hTheta)' * -y - log(1 - hTheta)'*(1 - y)) / m;
% add the regularization
J += sum(theta(2:end) .^ 2) * lambda /(2 * m);

grad = 1/m * X' * (hTheta - y); 
temp = theta;
temp(1) = 0;
grad = grad + lambda/m * temp;

% =============================================================

grad = grad(:);

end
