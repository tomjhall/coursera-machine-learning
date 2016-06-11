% ---------------------------------------
% Test 3a (Collaborative Filtering Cost):
% input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 0;
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
% output:
J =  311.63

if false
% ---------------------------------------
% Test 4a (Collaborative Filtering Gradient):
% input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 0;
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

% output:
% J =  311.63

% grad =
%   -16.1880
%   -23.5440
%    -5.1590
%   -14.9720
%   -21.4380
%   -30.4620
%    -6.5660
%   -19.5440
%    -3.4230
%    -7.0280
%    -3.4140
%   -12.2590
%   -16.0600
%    -9.7420


%---------------------------------------
%Test 5a (Regularized Cost):
%input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 6;
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

%output:
%J =  342.08
%
%---------------------------------------
%Test 6a (Gradient with regularization):
%input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 6;
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
% output:
J =  342.08

%grad =
%  -15.5880
%  -22.3440
%   -3.3590
%  -12.5720
%  -18.4380
%  -26.8620
%   -2.3660
%  -14.7440
%    1.9770
%   -1.0280
%    3.1860
%   -5.0590
%   -8.2600
%   -1.3420
%-----------
%Test 6b (a user with no reviews):
%input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 0; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 6;
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
% output:
J =  331.08

%grad =
%  -15.5880
%  -22.3440
%    1.8000
%  -12.5720
%  -18.4380
%  -26.8620
%    4.2000
%  -14.7440
%    1.9770
%   -1.0280
%    4.5930
%   -5.0590
%   -8.2600
%    1.9410
end
