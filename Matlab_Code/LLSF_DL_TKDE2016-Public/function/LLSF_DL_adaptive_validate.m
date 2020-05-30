
function [ BestParameter, BestResult ] = LLSF_DL_adaptive_validate( train_data, train_target, oldoptmParameter, modelparameter)
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% tune the best parameters for the model by 5-fold crossvalidation
% grid search : searching the best parameters for LLSF-DL on the training data by evaluating the performance of LLSF-DL in terms with 
% the combination of Accuracy, F1, Macro F1 and Micro F1. Of course, any evaluation metrics or the combination of them can be used.
% train_data   : n by d data matrix
% train_target : n by l label matrix
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    num_train         = size(train_data,1);
    randorder         = randperm(num_train);
    optmParameter     = oldoptmParameter;
    alpha_searchrange = modelparameter.alpha_searchrange;
    beta_searchrange  = modelparameter.beta_searchrange;
    gamma_searchrange = modelparameter.gamma_searchrange;
    rho_searchrange   = modelparameter.rho_searchrange;
    num_cv = 5;
    BestResult = zeros(15,1);
    index = 1;
    total = length(alpha_searchrange)*length(beta_searchrange)*length(gamma_searchrange)*length(rho_searchrange);
    
    fid = fopen('BestParameters.txt','wt');
    for i=1:length(alpha_searchrange) % alpha
        for j=1:length(beta_searchrange) % beta
            for k = 1:length(gamma_searchrange)
                for l = 1:length(rho_searchrange)
                    fprintf('\n-   %d-th/%d: searching parameter for LLSF-DL, alpha = %f, beta = %f, gamma = %f, and rho = %f',...
                            index, total, alpha_searchrange(i), beta_searchrange(j), gamma_searchrange(k),rho_searchrange(l));
                    optmParameter.alpha   = alpha_searchrange(i); % label correlation
                    optmParameter.beta    = beta_searchrange(j);  % sparsity of label specific features
                    optmParameter.gamma   = gamma_searchrange(k); % sparsity of label specific dependent labels
                    optmParameter.rho     = rho_searchrange(l);   % parameter for the initialization of W
                    optmParameter.maxIter           = 100;
                    optmParameter.minimumLossMargin = 0.1;
                    Result = zeros(15,1);

                    for cv = 1:num_cv
                        [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( train_data,train_target,randorder,cv,5);
                         model_LLSF_DL  = LLSF_DL( cv_train_data, cv_train_target,optmParameter); 
                        [~, cv_predict_target] = LLSF_TrainAndPredict(cv_train_data, cv_train_target,cv_test_data,optmParameter);
                        [Pre_Labels,Outputs] = LLSF_DL_Predict(model_LLSF_DL, cv_test_data, cv_predict_target, 3);
                        Result      = Result + EvaluationAll(Pre_Labels,Outputs,cv_test_target');
                    end
                    
                    Result = Result./num_cv;
                    r = IsBetterThanBefore(BestResult,Result);
                    if r == 1
                        BestResult = Result
                        BestParameter = optmParameter
                        
                        fprintf(fid, '\n-   %d-th/%d: search parameter for LLSF-DL, alpha = %f, beta = %f, gamma = %f, and rho = %f\n',...
                                index, total, alpha_searchrange(i), beta_searchrange(j), gamma_searchrange(k),rho_searchrange(l));
                        for aa = 1:15
                            fprintf(fid,'%.4f\n',BestResult(aa,1));
                        end
                    end
                    index = index + 1;
                end
            end % for k
        end % for j
    end
    fclose(fid);
end

% 1 HammingLoss
% 2 ExampleBasedAPCCuracy
% 3 ExampleBasedPrecision
% 4 ExampleBasedRecall
% 5 ExampleBasedFmeasure
% 6 SubsetAPCCuracy
% 7 LabelBasedAPCCuracy
% 8 LabelBasedPrecision
% 9 LabelBasedRecall
% 10 LabelBasedFmeasure
% 11 MicroF1Measure
% 12 Average_Precision
% 13 OneError
% 14 RankingLoss
% 15 Coverage
function r = IsBetterThanBefore(Result,CurrentResult)
    a = CurrentResult(2,1) + CurrentResult(5,1)  + CurrentResult(10,1) + CurrentResult(11,1);%+ CurrentResult(12,1) + CurrentResult(6,1);
    b = Result(2,1) + Result(5,1) + Result(10,1) + Result(11,1); % + Result(12,1)+ Result(6,1) ;

    if a > b
        r =1;
    else
        r = 0;
    end
end
