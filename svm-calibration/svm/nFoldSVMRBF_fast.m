function [model, cvaccuracy, yhat, testaccuracy] = nFoldSVMRBF_fast(traindata,testdata,nr_fold)
    disp([num2str(nr_fold),'-fold SVM model search with RBF kernel ']);
    logc_coarse = [-1:1:8];
    logg_coarse = [-1:1:8];
    
    params = getParameterList(logc_coarse, logg_coarse);
    [best_logc,best_logg,bestcv] = run_cv(traindata.X,traindata.y,nr_fold,params); 
    disp(['Best coarse cv ', num2str(2^best_logc),' ', num2str(2^best_logg),' ',  num2str(bestcv)]);
    
    % fine cross validation
    logc_fine = best_logc + [-1 : 0.25 : 1];
    logg_fine = best_logg + [-1 : 0.25 : 1];
    
    params = getParameterList(logc_fine, logg_fine);
    [best_logc,best_logg,bestcv] = run_cv(traindata.X,traindata.y,nr_fold,params); 
    disp(['Best fine cv ', num2str(2^best_logc),' ', num2str(2^best_logg),' ',  num2str(bestcv)]);
    cvaccuracy = bestcv;
    %bestc = 2^best_logc;
    %bestg = 2^best_logg;
    
    % train the model with best cross-valiation params
    %cmd = ['-q -s 0 -t 2 -c ', num2str(2^best_logc), ' -g ', num2str(2^best_logg)];
    %model = svmtrain(traindata.y,traindata.X,cmd);
	%[yhat,~,~] = svmpredict(testdata.y,testdata.X,model);
    %testaccuracy = sum(yhat == testdata.y)/length(testdata.y);
    param.gamma = 2^best_logg;
    tr_K = kernelmatrix('rbf',traindata.X,[],param);
    te_K = kernelmatrix('rbf',testdata.X,traindata.X,param);
    model = train(tr_K,traindata.y,2^best_logc);
    
	[yhat,testaccuracy] = test(model,te_K,testdata.y);
    
    disp(['Test accuracy with RBF kernel', num2str(testaccuracy)]);
    % store this as well
    model.best_logc = best_logc;
    model.best_logg = best_logg;
end


% function [bestc, bestg] = do_cv(X,y,nr_fold)
%     % coarse cross validation
%     logc_coarse = [-3:1:4];
%     logg_coarse = [-3:1:4];
% 
%     params = getParameterList(logc_coarse, logg_coarse);
%     [best_logc,best_logg,bestcv] = run_cv(X,y,nr_fold,params); 
%     disp(['Best coarse cv ', num2str(2^best_logc),' ', num2str(2^best_logg),' ',  num2str(bestcv)]);
% 
%     % fine cross validation
%     logc_fine = best_logc + [-1 : 0.025 : 1];
%     logg_fine = best_logg + [-1 : 0.025 : 1];
% 
%     params = getParameterList(logc_fine, logg_fine);
%     [best_logc,best_logg,bestcv] = run_cv(X,y,nr_fold,params); 
%     disp(['Best fine cv ', num2str(2^best_logc),' ', num2str(2^best_logg),' ',  num2str(bestcv)]);
% 
%     bestc = 2^best_logc;
%     bestg = 2^best_logg;
% end

function [best_logc,best_logg,bestcv] = run_cv(X,y,nr_fold,params)
    nrRun = length(params);
    %parfor i = 1 : nrRun % PARFOR is possible here!
    for i = 1 : nrRun % PARFOR is possible here!
    %for i = 1 : nrRun % PARFOR is possible here!
        %cmd = ['-q -s 0 -t 2 -c ', num2str(2^params{i}.logc), ' -g ', num2str(2^params{i}.logg)];
        %params{i}.cv_acc = get_cv_ac(X, y, cmd,nr_fold);
        params{i}.cv_acc = get_cv_ac(X, y,2^params{i}.logc,2^params{i}.logg,nr_fold);
        fprintf('Cross-validation %g %g %g\n', params{i}.logc, params{i}.logg, params{i}.cv_acc);
    end
    bestcv = 0;
    bestIndex = 0;
    for i = 1 : nrRun
        if(bestcv <=  params{i}.cv_acc),
            bestcv = params{i}.cv_acc; 
            bestIndex = i;
        end 
    end
    best_logc = params{bestIndex}.logc;
    best_logg = params{bestIndex}.logg;
end

function [ac] = get_cv_ac(x,y,C,gamma,nr_fold)
    len=length(y);
	rand_ind = randperm(len);
    %ac = zeros(nr_fold,1);
    param.gamma = gamma;
    K = kernelmatrix('rbf',x,[],param);

    yhat = zeros(len,1);
	
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
    ac = sum(yhat == y)/len;
end


function [model] = train(Ktr,ytr,C)
    numTrain = size(Ktr,1);
    Ktr = [(1:numTrain)' Ktr];
    model = svmtrain(ytr, Ktr, ['-q -t 4 -c ' num2str(C)]);
end

function [yhat,accuracy] = test(model,Kte,yte)
    numTest = size(Kte,1);
    Kte = [(1:numTest)' Kte];
    [yhat, accuracy, ~] = svmpredict(yte, Kte, model);
    accuracy = accuracy(1);
end


function [ PARAMETERS ] = getParameterList(logc, logg)
    pIdx=0;
    for i = 1:length(logc)
        for k = 1:length(logg)
            pIdx = pIdx+1;
            PARAMETERS{pIdx}.logc = logc(i);
            PARAMETERS{pIdx}.logg = logg(k);
            PARAMETERS{pIdx}.cv_acc = 0.0;
        end
    end
end