function [Pre_Labels,Outputs] = LLSF_DL_Predict(model_LLSFDL, cv_test_data, Pre_Labels, num_iteration)
% cv_test_data : n by d
% Pre_Labels: l by n

    optmParameter = model_LLSFDL.optmParameter ;
    
    for i =1:num_iteration
        Outputs       = (optmParameter.thetax*cv_test_data*model_LLSFDL.W_x + optmParameter.thetay*Pre_Labels'*model_LLSFDL.W_y);
        Outputs       = Outputs';
        
        Pre_Labels  = round(Outputs);
        Pre_Labels  = (Pre_Labels>= 0.5);
        Pre_Labels  = double(Pre_Labels);
    end
end

