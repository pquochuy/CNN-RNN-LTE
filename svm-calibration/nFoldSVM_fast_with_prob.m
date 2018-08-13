function [model, cvaccuracy, yhat, testaccuracy, prob] = nFoldSVM_fast_with_prob(tr_data,te_data,dist_func,nr_fold)

Ntest = numel(te_data.y);
Ntrain = numel(tr_data.y);

switch dist_func
	case {'linear','chi_square','histogram_intersection'}
        tr_K = kernelmatrix(dist_func,tr_data.X,[],[]);
        te_K = kernelmatrix(dist_func,te_data.X,tr_data.X,[]);
    case {'chi_square_rbf'}
        tr_K = kernelmatrix('chi_square',tr_data.X,[],[]);
        meanK = sum(tr_K(:))/(Ntrain*(Ntrain - 1));
        tr_K = exp(-tr_K/meanK);
        te_K = pdist2('chi_square',te_data.X,tr_data.X,[]);
        te_K = exp(-te_K/meanK);
	case {'histogram_intersection_rbf'}
        tr_K = kernelmatrix('histogram_intersection',tr_data.X,[],[]);
        meanK = sum(tr_K(:))/(Ntrain*(Ntrain - 1));
        tr_K = exp(-tr_K/meanK);
        te_K = pdist2('histogram_intersection',te_data.X,tr_data.X,[]);
        te_K = exp(-te_K/meanK);
    otherwise
        error(['Error pdist2 - distance function: ' dist_func]);
end



[model, cvaccuracy, yhat, testaccuracy, prob] = nFoldSVMMatrix(tr_K,tr_data.y,te_K,te_data.y,nr_fold);
end


function [model, cvaccuracy, yhat, testaccuracy, prob] = nFoldSVMMatrix(tr_K,tr_y,te_K,te_y,nr_fold)

%logc_coarse = [-3:1:8];
logc_coarse = [-3:1:8];
error_coarse = zeros(size(logc_coarse));
logc_fine_template = [-1 : 0.25 : 1];

% k-fold cross validation on C
%parfor c = 1 : length(logc_coarse)
for c = 1 : length(logc_coarse) % without parloop is faster due to overhead
    % use this to make sure we are searching for the right things
    trainAndTest(tr_K,tr_y,te_K,te_y,2^logc_coarse(c));
    error_coarse(c) = run_cv(tr_K,tr_y,2^logc_coarse(c),nr_fold);
end

% find minimum error rate in coarse search
[val,ind] = min(error_coarse);
ind = find(error_coarse==val,1,'last');

disp(['Best coarse xValidation at log10(C) ', num2str(logc_coarse(ind)), ' error: ',  num2str(val)]);

% fine search parameter
logc_fine = logc_fine_template + logc_coarse(ind);
error_fine = zeros(size(logc_fine));
%parfor c = 1 : length(logc_fine)
for c = 1 : length(logc_fine)
    trainAndTest(tr_K,tr_y,te_K,te_y,2^logc_fine(c));
    error_fine(c) = run_cv(tr_K,tr_y,2^logc_fine(c),nr_fold);
end
[val,ind] = min(error_fine);
ind = find(error_fine==val,1,'last');

best_logc = logc_fine(ind);
disp(['Best fine xValidation at log10(C) ', num2str(logc_fine(ind)), ' error: '  num2str(val)]);
cvaccuracy = 1.0 - val;

% train and test the final model
model = train(tr_K,tr_y,2^best_logc);
[yhat,testaccuracy,prob] = test(model,te_K,te_y);
end

function [error] = run_cv(K,y,C,nr_fold)
    len = length(y);
    rand_ind = randperm(len);
    yhat = zeros(size(y));
    
    for i=1:nr_fold % Cross training : folding
        test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
        train_ind = [1:len]';
        train_ind(test_ind) = [];
        
        Ktr = K;
        ytr = y;
        Ktr(test_ind,:) = [];
        Ktr(:,test_ind) = [];
        ytr(test_ind) = [];
        
        Kte = K(test_ind,:);
        Kte(:,test_ind) = [];
        yte = y(test_ind);
        model_i = train(Ktr,ytr,C);
        [yhat(test_ind),~] = test(model_i,Kte,yte);
    end
    error = sum(yhat ~= y)/len;
end

function trainAndTest(Ktr,ytr,Kte,yte,C)
    numTrain = size(Ktr,1);
	numTest = size(Kte,1);
    Ktr = [(1:numTrain)' Ktr];
    Kte = [(1:numTest)' Kte];
    model = svmtrain(ytr, Ktr, ['-q -t 4 -c ' num2str(C),' -b 1']);
    [~, accuracy, ~] = svmpredict(yte, Kte, model, '-b 1');
    disp(['DEBUG: with logC = ', num2str(log10(C)),': ', num2str(accuracy(1))]);
end

function [model] = train(Ktr,ytr,C)
    numTrain = size(Ktr,1);
    Ktr = [(1:numTrain)' Ktr];
    model = svmtrain(ytr, Ktr, ['-q -t 4 -c ' num2str(C),' -b 1']);
end

function [yhat,accuracy,prob] = test(model,Kte,yte)
    numTest = size(Kte,1);
    Kte = [(1:numTest)' Kte];
    [yhat, accuracy, prob] = svmpredict(yte, Kte, model, '-b 1');
    accuracy = accuracy(1);
end