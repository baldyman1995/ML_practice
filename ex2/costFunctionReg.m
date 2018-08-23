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

for ii = 1:m
	J = J + (- y(ii) * log(1 / (1 + exp( - (X(ii,:) * theta)))) - (1 - y(ii)) * log(1 - 1 / (1 + exp( - (X(ii,:) * theta)))));
end

J = J/m + (lambda * ((theta' * theta) - theta(1,1) * theta(1,1)))/(2*m);


for jj = 1:length(theta)

	for ii = 1:m
		grad(jj) = grad(jj) + (1 / (1 + exp( - (X(ii,:) * theta))) - y(ii)) * X(ii,jj);
	end

	if jj == 1
		grad(jj) = grad(jj) / m;
	else
		grad(jj) = grad(jj) / m + (lambda * theta(jj))/m;
	end
	
end




% =============================================================

end
