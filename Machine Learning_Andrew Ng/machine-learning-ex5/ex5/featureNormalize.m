function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);                   %����ÿ�о�ֵ
X_norm = bsxfun(@minus, X, mu); %��ȥ������ֵ

sigma = std(X_norm);            %���ÿ�б�׼��
X_norm = bsxfun(@rdivide, X_norm, sigma); %��ǰ�沽��һ�𣬽��й�һ��


% ============================================================

end
