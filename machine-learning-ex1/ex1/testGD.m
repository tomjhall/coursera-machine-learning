function [theta, J_history] = testGD()
  % testX = [1 1; 1 2; 1 3];
  % testy = [1; 2; 3];
  testX = [1 1; 1 2; 1 3];
  testy = [3; 2; 1];
  % testX = [1 1; 1 2];
  % testy = [1; 0];
  testTheta = [0 0];

  [theta, J_history] = gradientDescent(testX,testy,testTheta',0.1,100);

  
end
