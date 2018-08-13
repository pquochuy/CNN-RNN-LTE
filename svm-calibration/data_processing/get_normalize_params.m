function [minimums, ranges] = get_normalize_params(X)
    minimums = min(X, [], 1);
    ranges = max(X, [], 1) - minimums;
end