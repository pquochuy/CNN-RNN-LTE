function [model, cvaccuracy, yhat, testaccuracy] = nFoldSVM(traindata,testdata,dist_func,nr_fold)
disp([num2str(nr_fold),'-fold SVM model search with ' func2str(dist_func) ' kernel ']);

logc_coarse = [-3:1:8];
error_coarse = zeros(size(logc_coarse));
logc_fine_template = [-1 : 0.25 : 1];

% k-fold cross validation on C
parfor c = 1 : length(logc_coarse)
    % use this to make sure we are searching for the right things
    trainAndTest(traindata,testdata,2^logc_coarse(c),dist_func);
    %error_coarse(c) = leave1OutX(traindata,2^logc_coarse(c),dist_func);
    error_coarse(c) = run_cv(traindata,2^logc_coarse(c),dist_func,nr_fold);
end

% find minimum error rate in coarse search
[val,ind] = min(error_coarse);
ind = find(error_coarse==val,1,'last');

disp(['Best coarse xValidation at log10(C) ', num2str(logc_coarse(ind)), ' error: ',  num2str(val)]);

% fine search parameter
logc_fine = logc_fine_template + logc_coarse(ind);
error_fine = zeros(size(logc_fine));
parfor c = 1 : length(logc_fine)
    trainAndTest(traindata,testdata,2^logc_fine(c),dist_func);
    %error_fine(c) = leave1OutX(traindata,2^logc_fine(c),dist_func);
    error_fine(c) = run_cv(traindata,2^logc_fine(c),dist_func,nr_fold);
end
[val,ind] = min(error_fine);
ind = find(error_fine==val,1,'last');

best_logc = logc_fine(ind);
disp(['Best fine xValidation at log10(C) ', num2str(logc_fine(ind)), ' error: '  num2str(val)]);
cvaccuracy = 1.0 - val;

% train and test the final model
model = train(traindata.X,traindata.y,2^best_logc,dist_func);
[yhat,testaccuracy] = test(model,traindata.X,traindata.y,testdata.X,testdata.y,dist_func);
end

function [error] = run_cv(data,C,dist_func,nr_fold)
    len=length(data.y);
	rand_ind = randperm(len);
    
    yhat = zeros(len,1);
    
    for i=1:nr_fold % Cross training : folding
      test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
      train_ind = [1:len]';
      train_ind(test_ind) = [];
      
      model_i = train(data.X(train_ind,:),data.y(train_ind),C,dist_func);
      [pred,~] = test(model_i,data.X(train_ind,:),data.y(train_ind),...
          data.X(test_ind,:),data.y(test_ind),dist_func);
      yhat(test_ind) = pred;
    end
    error = sum(yhat ~= data.y)/len;
end

function trainAndTest(traindata,testdata,C,dist_func)
    switch func2str(dist_func)
        case {0,'linear'}
            model = svmtrain(traindata.y, traindata.X, ['-q -t 0 -c ' num2str(C)]);
            [yhat, accuracy, ~] = svmpredict(testdata.y, testdata.X, model);
            disp(['DEBUG: Training ', func2str(dist_func),' with logC = ', num2str(log10(C)),': ', num2str(accuracy(1))]);
        case {'chi_square_statistics_fast','histogram_intersection'}
            numTrain = size(traindata.X,1);
            numTest = size(testdata.X,1);
            % calculate pairwise distance 
            K = pdist2(traindata.X,traindata.X,dist_func);
            % calculate similarity (kernel function) 
            K = ones(size(K)) - K;
            K = [(1:numTrain)' K];

            KK = pdist2(testdata.X,traindata.X,dist_func);
            KK = ones(size(KK)) - KK;
            KK = [(1:numTest)' KK];

            % BUILD NEW MODEL - ADD YOUR MODEL BUILDING CODE HERE...
            model = svmtrain(traindata.y, K, ['-q -t 4 -c ' num2str(C)]);

            % EVALUATE WITH TEST DATA - ADD YOUR MODEL EVALUATION CODE HERE
            [yhat, accuracy, ~] = svmpredict(testdata.y, KK, model);
            disp(['DEBUG: Training ', func2str(dist_func),' with logC = ', num2str(log10(C)),': ', num2str(accuracy(1))]);
        otherwise
            error(['Error pdist2 - distance function: ' func2str(dist_func)]);
    end
end

function [model] = train(Xtr,ytr,C,dist_func)
    switch func2str(dist_func)
        case {0,'linear'}
            model = svmtrain(ytr, Xtr, ['-q -t 0 -c ' num2str(C)]);
        case {'chi_square_statistics_fast','histogram_intersection'}
            numTrain = size(Xtr,1);
            % calculate pairwise distance 
            K = pdist2(Xtr,Xtr,dist_func);
            % calculate similarity (kernel function) 
            K = ones(size(K)) - K;
            K = [(1:numTrain)' K];
            % BUILD NEW MODEL - ADD YOUR MODEL BUILDING CODE HERE...
            model = svmtrain(ytr, K, ['-q -t 4 -c ' num2str(C)]);
        otherwise
            error(['Error pdist2 - distance function: ' func2str(dist_func)]);
    end
end

function [yhat,accuracy] = test(model,Xtr,ytr,Xte,yte,dist_func)
    switch func2str(dist_func)
        case {0,'linear'}
            [yhat, accuracy, ~] = svmpredict(yte, Xte, model);
            accuracy = accuracy(1);
        case {'chi_square_statistics_fast','histogram_intersection'}
            numTest = size(Xte,1);
            KK = pdist2(Xte,Xtr,dist_func);
            KK = ones(size(KK)) - KK;
            KK = [(1:numTest)' KK];
            % EVALUATE WITH TEST DATA - ADD YOUR MODEL EVALUATION CODE HERE
            [yhat, accuracy, ~] = svmpredict(yte, KK, model);
            accuracy = accuracy(1);
        otherwise
            error(['Error pdist2 - distance function: ' func2str(dist_func)]);
    end
end