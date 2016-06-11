function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% num_iters = 2;
% X = [1 1; 1 2; 1 3];
% y = [1;2;3];

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


    % multivariant_theta = [1;theta]
    n = length(theta);
    theta_save = zeros(n, 1);

    % for parameter = 1:n
      % theta_save(parameter) = theta(parameter) - (alpha * computeCost(X, y, theta));

      for parameter = 1:n
        sum = 0;
        for i = 1:m
         % printf('n is %d',n)
         % printf('m is %d',m)
          printf('h = %f * %f + %f + %f\n',X(i,1),theta(1),X(i,2),theta(2))
          h = X(i,1) * theta(1) + X(i,2) * theta(2)
	  h = h - y(i)
	  h = h * X(i,parameter)

          sum += X(i,:) * theta - y(i);
        end

	printf('sum of (%d) is %f\n',parameter,sum)

	% print('J(%f,%f) = %f\n',theta(1),theta(2),sum);
        % sum += ((X * theta) - y) * X(m,n)
	% printf('sum is %f\n',sum)

	% printf('theta(%d) was %f but now is %f\n',parameter,theta(parameter),theta(parameter) - (alpha/m * sum))

	theta_save(parameter) = theta(parameter) - (alpha/m * sum);
      end

      theta = theta_save;

      % h is theta1

      % printf('Theta(%d) is %f\n',parameter,theta(parameter))

      % theta(1)
      % theta(2)

      % X(1,1)
      % X(1,2)

      % H = ((X * theta) - y) * X()

      % multivariant_X = [ones(size(X,1), 1) X]

      % H = [ones(size(X,1), 1) X] * multivariant_theta

      % H = X * theta;

      % printf('Parameter(%d) was %f, now is %f\n',parameter,theta(parameter),theta_save(parameter))


      %theta_save(parameter) = theta(parameter) - (alpha *  
      % printf('Parameter(%d) was %f, now is %f\n',parameter,theta(parameter),theta_save(parameter))
    %end

    % theta_save = multivariant_theta(2:n,:);
    % printf('cost was %f, now is %f\n',computeCost(X, y, theta), computeCost(X, y, theta_save));
    % theta = multivariant_theta(2:n,:);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

% figure; % open a new figure window
% plot(J_history);

end
