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
for i=1:m
  currentX=X(i,:);
  a=currentX*theta;
  currentY=y(i,1);
  J=J-currentY*log(sigmoid(a))-(1-currentY)*log(1-sigmoid(a));
  for k=1:size(grad)
    grad(k)=grad(k)+(sigmoid(a)-currentY)*currentX(1,k);
  end 
end
J=(J+lambda*0.5*sum(theta(2:end).^2))/m;
grad(2:end)=(grad(2:end)+lambda*(theta(2:end)))/m;
grad(1)=grad(1)/m;





% =============================================================

end
