function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
theta_t=theta;
theta_t(1)=0;
J=sum((X*theta-y).^2)/2/m+lambda*sum(theta_t.^2)/2/m; %不要忘记偏置对应的theta不计算


%矩阵形式

grad=sum(repmat(X*theta-y,1,size(X,2)).*X,1)'/m+lambda*theta_t/m;

%单维计算
% grad(1)=sum((X*theta-y).*X(:,1))/m;
% for i=2:size(X,2)
%     grad(i)=sum((X*theta-y).*X(:,i))/m+lambda*theta(i)/m;
% end













% =========================================================================

grad = grad(:);

end
