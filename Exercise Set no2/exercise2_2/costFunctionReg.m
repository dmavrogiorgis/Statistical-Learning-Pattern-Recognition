function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sig = sigmoid(X*theta);
temp_log(:,1) = log(sig);
temp_log(:,2) = log(1-sig);
temp_y(:,1) = y;
temp_y(:,2) = 1-y;
temp = temp_log.*temp_y;

theta(1) = 0;

J = (1/m)*(-sum(temp(:,1))-sum(temp(:,2))) + (lambda/(2*m))*sum(theta.^2);

grad = (1/m)*X'*(sig-y) + (lambda/m)*theta;

% =============================================================

end
