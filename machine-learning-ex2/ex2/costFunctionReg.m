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

for i=1:1:m
    
    z= X(i,:)*theta;
    
    J= J +( -1*y(i)*log(sigmoid(z)) - (1-y(i))*log(1-sigmoid(z)) );
    
    grad(1,1)= grad(1,1) + (1/m)*(sigmoid(z) - y(i)) .* X(i,1);
    for j=2:1:length(grad)
        grad(j,1)= grad(j,1) + (1/m)*(sigmoid(z) - y(i)) .* X(i,j);
    end
end

J= (1/m)*J + (lambda/(2*m))*sum(theta(2:end,1).^2);

for j=2:1:length(grad)
    grad(j,1)= grad(j,1) + (lambda/m)*theta(j);
end




% =============================================================

end
