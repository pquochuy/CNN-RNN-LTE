% get whitening nomalization parameters to have zero mean and unity variance for each feature
function [X2] = whitening(X, mu, sigma)
%		[mu, sigma] = get_whitening_params(X)
%		X2 = whitening(X, mu, sigma)

[N, ~] = size(X);
X2 = (X - repmat(mu, N, 1)) ./ repmat(sigma, N, 1);

end