%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is an examplar file on how the LLSF_DL [2] program could be used.
% 
% [1] J. Huang, G.-R Li, Q.-M. Huang and X.-D. Wu. Learning Label Specific Features for Multi-Label Classifcation. 
%     In: Proceedings of the International Conference on Data Mining, 2015.
% [2] J. Huang, G.-R Li, Q.-M. Huang and X.-D. Wu. Learning Label-Specific Features and Class-Dependent Labels 
%     for Multi-Label Classification, To appear in TKDE, 2016.
% 
% Please feel free to contact me (huangjun.cs@gmail.com), if you have any problem about this programme.
% http://www.escience.cn/people/huangjun/index.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off %#ok<WNOFF>
addpath(genpath('.'));
clc
clear all
load CAL500.mat;

starttime = datestr(now,0);
deleteData  = 0;

[optmParameter, modelparameter] =  initialization;

model_LLSFDL.optmParameter = optmParameter;
model_LLSFDL.modelparameter = modelparameter;

%% 
time_tune = zeros(1,modelparameter.cv_num);
time_train = zeros(1,modelparameter.cv_num);
time_test = zeros(1,modelparameter.cv_num);

%% Train and Test
if modelparameter.crossvalidation==0 
    
else
%% cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if exist('train_data','var')==1
        data    = [train_data;test_data];
        target  = [train_target,test_target];
        if deleteData == 1
            clear train_data test_data train_target test_target
        end
    end
    data      = double (data);
    num_data  = size(data,1);
    temp_data = data + eps;
    
    if modelparameter.L2Norm == 1
        temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
        if sum(sum(isnan(temp_data)))>0
            temp_data = data+eps;
            temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
        end
    end
    temp_data = [temp_data,ones(num_data,1)];
    if deleteData == 1
        clear data;
    end
    if modelparameter.tuneparameter==1
        randorder = 1:num_data; % fixed splition of training and test parts
    else
        randorder = randperm(num_data);
    end

    cvResult  = zeros(15,modelparameter.cv_num);
    TunedParameters = cell(1,modelparameter.cv_num);
    Wset = cell(1,modelparameter.cv_num);
    for j = 1:modelparameter.cv_num
        fprintf('- Cross Validation - %d/%d  ', j, modelparameter.cv_num);
        [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( temp_data,target',randorder,j,modelparameter.cv_num );
        
        if modelparameter.searchbestparas == 1
            fprintf('\n-  parameterization for LLSF-DL by cross validation on the training data');
            tic
            if (optmParameter.tuneParaOneTime == 1) && (exist('BestResult','var')==0)
                [ BestParameter, BestResult] = LLSF_DL_adaptive_validate(cv_train_data, cv_train_target,optmParameter,modelparameter);
            elseif optmParameter.tuneParaOneTime == 0
                [ BestParameter, BestResult] = LLSF_DL_adaptive_validate(cv_train_data, cv_train_target,optmParameter,modelparameter);
            end
            time_tune(1,j) = toc

            tic
            [W,loss]  = LLSF_DL( cv_train_data, cv_train_target,BestParameter); 
            [~, cv_predict_target] = LLSF_TrainAndPredict(cv_train_data, cv_train_target,cv_test_data,optmParameter, 0);
            time_train(1,j) = toc;
            TunedParameters{1,j} = BestParameter;
            
        else % fixed parametrs
            tic
            W  = LLSF_DL( cv_train_data, cv_train_target,optmParameter); 
            [~, cv_predict_target] = LLSF_TrainAndPredict(cv_train_data, cv_train_target,cv_test_data,optmParameter);
            time_train(1,j) = toc;
        end
        Wset{1,j} = W;
       
       %% prediction and evaluation
        tic
        [Pre_Labels,Outputs] = LLSF_DL_Predict(W, cv_test_data, cv_predict_target, 3);
        cvResult(:,j) = EvaluationAll(Pre_Labels,Outputs,cv_test_target')
        time_test(1,j) = toc;
        
    end
    
    Avg_Result      = zeros(15,2);
    Avg_Result(:,1) = mean(cvResult,2);
    Avg_Result(:,2) = std(cvResult,1,2);
    PrintResults(Avg_Result);
    
    model_LLSFDL.cvTuneTime   = time_tune;
    model_LLSFDL.cvTrainTime  = time_train;
    model_LLSFDL.cvTestTime   = time_test;
    model_LLSFDL.avgTuneTime  = mean(time_tune);
    model_LLSFDL.avgTrainTime = mean(time_train);
    model_LLSFDL.avgTestTime  = mean(time_test);
    
    model_LLSFDL.randorder = randorder;
    model_LLSFDL.avgResult = Avg_Result;
    model_LLSFDL.cvResult  = cvResult;
    model_LLSFDL.WSet = Wset;
    model_LLSFDL.cvTunedParameters = TunedParameters;
end

endtime = datestr(now,0);

