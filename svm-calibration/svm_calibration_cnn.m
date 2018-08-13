% without probability --> this only need for multistream fusion

function svm_calibration_cnn()
	Nfold = 20;
    
    %parfor fold = 1 : Nfold
    for fold = 1 : Nfold
        disp(['Fold ', num2str(fold)])
         train_and_save(fold);
    end
    
end

function ret = train_and_save(fold)
    mat_path = ['../cnn_tensorflow_output_path/fold',num2str(fold),'/'];
    
    load([mat_path, 'train_feat.mat']); % extract features of training data from RNN-LTE
    load([mat_path, 'test_feat.mat']); % extract features of testing data from RNN-LTE
    
    % remove all zeros features
    sum_feat = sum(train_data);
    train_data(:,sum_feat == 0) = [];
    test_data(:,sum_feat == 0) = [];
    
    % remove all zeros features
    sum_feat = sum(test_data);
    train_data(:,sum_feat == 0) = [];
    test_data(:,sum_feat == 0) = [];
    
    % normalization to [0,1]
    [minX, rangeX] = get_normalize_params(train_data);
    % normalize data to [0,1]
    train_data = normalize_data(train_data,minX,rangeX,0,1);
    test_data = normalize_data(test_data,minX,rangeX,0,1);

    
    % clipping
    for i = 1 : size(test_data,1)
        ind = test_data(i,:) < 0.0;
        test_data(i,ind) = 0.0;
        ind = test_data(i,:) > 1.0;
        test_data(i,ind) = 1.0;
    end
    
    traindata.X = train_data;
    traindata.y = double(train_label);
    testdata.X = test_data;
    testdata.y = double(test_label);
    dist_func = @linear;
    %[model, ~, yhat, acc,prob] = nFoldSVM_fast_with_prob(traindata, testdata, dist_func,10);
    [model, ~, yhat, acc,prob] = nFoldSVM_with_prob(traindata,testdata,dist_func,10);
    %[model, ~, yhat, acc] = nFoldSVM_fast(traindata, testdata, dist_func,10);
    ret.model_lin = model;
    ret.yhat_lin = yhat;
    ret.acc_lin = acc;
    ret.prob_lin = prob;
    save([mat_path,'ret.mat'], 'ret');
end

