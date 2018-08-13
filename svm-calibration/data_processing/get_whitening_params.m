% get whitening nomalization parameters to have zero mean and unity variance for each feature
function [mu, sigma] = get_whitening_params(X)
%		[mu, sigma] = get_whitening_params(X)
%		X2 = whitening(X, mu, sigma)
%

    mu = mean(X, 1);
    sigma = sqrt(var(X, 0, 1));

end