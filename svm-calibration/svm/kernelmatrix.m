% Precompute kernel matrix to speed up cross-validation using libsvm
% Inputs:
%       ker:            'linear','poly','rbf'
%       X1:             PxM data matrix, P samples, M features
%       X2:             QxM data matrix, Q samples, M features
%       param.gamma:    the RBF kernel (K(x1,x2) = exp(-gamma||x1-x2||^2)
%       param.b:        bias in the polinomial kernel
%       param.d:        degree of the polynomial kernel
%
% Output:
%       K: kernel matrix
%
% https://code.google.com/r/sandyshenli-altoolbox/source/browse/kernelmatrix.m

function K = kernelmatrix(ker,X1,X2,param)

switch ker
    case 'linear'
        if (exist('X2','var') && ~isempty(X2))
            K = X1 * X2';
        else
            K = X1 * X1';
        end

    case 'poly'
        if (exist('X2','var') && ~isempty(X2))
            K = (X1 * X2' + param.b).^param.d;
        else
            K = (X1 * X1' + param.b).^param.d;
        end

    % http://stackoverflow.com/questions/21826439/libsvm-with-precomputed-kernel-how-do-i-compute-the-classification-scores
    case 'rbf'
        % RBF kernel: exp(-gamma*|X1-X2|^2)
        rbf = @(X,Y) exp(-param.gamma .* pdist2(X,Y,'euclidean').^2);
        if (~exist('X2','var') || isempty(X2));
            K = rbf(X1,X1);
        else
            K = rbf(X1,X2);
        end;

    case 'chi_square'
        if (~exist('X2','var') || isempty(X2));
            K = pdist2(X1,X1,@chi_square_statistics_fast);
            K = ones(size(K)) - K;
        else
            K = pdist2(X1,X2,@chi_square_statistics_fast);
            K = ones(size(K)) - K;
        end
        
    case 'histogram_intersection'
        if (~exist('X2','var') || isempty(X2));
            K = pdist2(X1,X1,@histogram_intersection);
            K = ones(size(K)) - K;
        else
            K = pdist2(X1,X2,@histogram_intersection);
            K = ones(size(K)) - K;
        end
        
        
%    Will be handle seperately
%     case 'chi_square_rbf'
%         if (~exist('X2','var') || isempty(X2));
%             K = pdist2(X1,X1,@chi_square_statistics_fast);
%         else
%             K = pdist2(X1,X2,@chi_square_statistics_fast);
%         end
%         % compute mean distance, and normalize
%         if(~isempty(param) && ~isfield(param,'mean_distance'))
%             meanK = sum(K(:))/(size(K,1)*(size(K,2) - 1));
%         else
%             meanK = param.mean_distance;
%         end
%         K = exp(-K/meanK);
%         
%     case 'histogram_intersection_rbf'
%         if (~exist('X2','var') || isempty(X2));
%             K = pdist2(X1,X1,@histogram_intersection);
%         else
%             K = pdist2(X1,X2,@histogram_intersection);
%         end
%         % compute mean distance, and normalize
%         if(~isempty(param) && ~isfield(param,'mean_distance'))
%             meanK = sum(K(:))/(size(K,1)*(size(K,2) - 1));
%         else
%             meanK = param.mean_distance;
%         end
%         K = exp(-K/meanK);
    otherwise
        error(['Unsupported kernel ' ker])
end