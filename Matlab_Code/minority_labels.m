% for minority labels only
clear
tic
addpath(genpath('.'));
datasets = [{'datasets/genbase'};{'datasets/emotions'};{'datasets/rcv1-sample1'};{'datasets/recreation'}];
datasetNo =1;
filename1 = datasets{datasetNo};
load(filename1);
ttlFold = 5;
%% Test Result

maxSelfIter    = 3;
ttlEva         = 15;
maxIter        = 100;
Avg_LLSF_Self_Smote   = zeros(ttlEva,maxSelfIter);
Result_LLSF_SMOTE{ttlFold} = zeros(ttlEva,maxSelfIter);
optmParameter                   = struct;
optmParameter.maxIter           = maxIter;
optmParameter.minimumLossMargin = 0.001;
optmParameter.bQuiet            = 1;
optmParameter.alpha             = 4^-3;      % label correlation
optmParameter.beta              = 4^-2;      % sparsity of label specific features
optmParameter.gamma             = 4^-1;      % sparsity of label specific dependent labels
optmParameter.rho               = 0.1;
optmParameter.thetax            = 1;
optmParameter.thetay            = 1;
%      Parameters        : alpha, beta, gamma, rho
%      for emotions      : 4^{5, 3, 3}, 0.1
%      for rcv1subset1   : 4^{5, 3, 3}, 1
%      for genbase       : 4^{-3,-2,-1}, 0.1
%      for recreation    : 4^{6, 4, 5},1
no_fold = 5;
dataX = [X;Xt];
dataY = [Y;Yt];
N          = size(dataY,1);
rand_idx   = randperm(N);
partationData    = kfoldpartation( dataX, dataY, no_fold,rand_idx);
%% SELF LEARN
for runNo=1:ttlFold
    X   =   full(partationData{runNo}.X);
    Y   =   full(partationData{runNo}.Y);
    Xt  =   full(partationData{runNo}.Xt);
    Yt  =   full(partationData{runNo}.Yt);
    [IR_label, meanir]=Imbalance_ratio(Y);
    [~,numL]=size(Y);
    minorityL = IR_label>meanir;
    
    Y_new   =[];
    Yt_new  =[];
    for c=1:numL
        if minorityL(c) % && sum(dataY(:,c),1)>2
            Y_new   =   [Y_new,Y(:,c)];
            Yt_new  =   [Yt_new,Yt(:,c)];
        end
    end
    Y=Y_new;
    Yt =Yt_new;
    clear Yt_new Y_new ;
    
    for selfIterNo = 1: maxSelfIter
        W = LLSF_DL( X, Y, optmParameter);
        %LLSF predict and train
        [Outputs_llsf, predict_Label] = LLSF_TrainAndPredict(X, Y,Xt,optmParameter);
        % prediction and evaluation
        [Pre_Labels,Outputs_DL] = LLSF_DL_Predict(W, Xt, predict_Label, 3);
        
        Result_LLSF_SMOTE{runNo}(:,selfIterNo) = EvaluationAll(predict_Label,Yt',Outputs_llsf);
        
        Xnew=[];
        Ynew=[];
        train_data=[];
        train_label=[];
        for L=1:size(Y,2)
            [train_data, train_label]=MLSMOTE(X,L,Y,a);
            Xnew=[Xnew;train_data];
            Ynew=[Ynew;train_label];
        end
        
        newIDXToRetain =  sum(Ynew.* repmat(ones(1,size(Ynew,2)),size(Xnew,1),1),2) >0;
        X = [X;Xnew(newIDXToRetain,:)];
        Y = [Y;Ynew(newIDXToRetain,:)];
        
    end
    fprintf(" Run completed %d \n",runNo);
end


for runNo=1:ttlFold
    Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote + Result_LLSF_SMOTE{runNo};
end

Avg_LLSF_Self_Smote  = Avg_LLSF_Self_Smote ./ ttlFold;
toc
