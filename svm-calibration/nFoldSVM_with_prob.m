function [model, cvaccuracy, yhat, testaccuracy, prob] = nFoldSVM_with_prob(traindata,testdata,dist_func,nr_fold)
disp([num2str(nr_fold),'-fold SVM model search with ' func2str(dist_func) ' kernel ']);

cvaccuracy = 1.0;
best_logc = 0.0;
% train and test the final model
model = train(traindata.X,traindata.y,2^best_logc,dist_func);
%model = train(traindata.X,traindata.y,best_logc,dist_func);
[yhat,testaccuracy,prob] = test(model,traindata.X,traindata.y,testdata.X,testdata.y,dist_func);
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
            model = svmtrain(traindata.y, traindata.X, ['-q -t 0 -c ' num2str(C),' -b 1']);
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
            model = svmtrain(traindata.y, K, ['-q -t 4 -c ' num2str(C),' -b 1']);

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
            model = svmtrain(ytr, Xtr, ['-q -t 0 -c ' num2str(C),' -b 1']);
        case {'chi_square_statistics_fast','histogram_intersection'}
            numTrain = size(Xtr,1);
            % calculate pairwise distance 
            K = pdist2(Xtr,Xtr,dist_func);
            % calculate similarity (kernel function) 
            K = ones(size(K)) - K;
            K = [(1:numTrain)' K];
            % BUILD NEW MODEL - ADD YOUR MODEL BUILDING CODE HERE...
            model = svmtrain(ytr, K, ['-q -t 4 -c ' num2str(C),' -b 1']);
        otherwise
            error(['Error pdist2 - distance function: ' func2str(dist_func)]);
    end
end

function [yhat,accuracy,prob] = test(model,Xtr,ytr,Xte,yte,dist_func)
    switch func2str(dist_func)
        case {0,'linear'}
            [yhat, accuracy, prob] = svmpredict(yte, Xte, model,' -b 1');
            accuracy = accuracy(1);
        case {'chi_square_statistics_fast','histogram_intersection'}
            numTest = size(Xte,1);
            KK = pdist2(Xte,Xtr,dist_func);
            KK = ones(size(KK)) - KK;
            KK = [(1:numTest)' KK];
            % EVALUATE WITH TEST DATA - ADD YOUR MODEL EVALUATION CODE HERE
            [yhat, accuracy, prob] = svmpredict(yte, KK, model,' -b 1');
            accuracy = accuracy(1);
        otherwise
            error(['Error pdist2 - distance function: ' func2str(dist_func)]);
    end
end