function [X_norm] = normalize_data(X,minimums,ranges,rmin,rmax)
    % normalize to range [0,1]
    X_norm = (X - repmat(minimums, size(X, 1), 1)) ./ repmat(ranges, size(X, 1), 1);
    % normalize to arbitray range [rmin,rmax]
    X_norm = (X_norm*(rmax - rmin)) + rmin;
end