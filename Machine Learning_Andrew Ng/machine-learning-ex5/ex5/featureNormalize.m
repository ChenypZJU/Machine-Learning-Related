function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);                   %返回每列均值
X_norm = bsxfun(@minus, X, mu); %减去样本均值

sigma = std(X_norm);            %获得每列标准差
X_norm = bsxfun(@rdivide, X_norm, sigma); %与前面步骤一起，进行归一化


% ============================================================

end
