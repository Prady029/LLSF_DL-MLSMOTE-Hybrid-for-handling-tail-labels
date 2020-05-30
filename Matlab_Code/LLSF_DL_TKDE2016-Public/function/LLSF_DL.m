
function model_LLSFDL = LLSF_DL( X, Y, optmParameter, YP)
% This function is designed to Learn label-Specific Features and Class-Dependent Labels for Multi-Label Classification
% 
%    Syntax
%
%       [model_LLSFDL] = LLSF_DL( X, Y, optmParameter)
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       YP              - a n by l label matrix which is predicted for the training data by another classifier, such as LLSF
%       optmParameter   - the optimization parameters for LLSF-DL, a struct variable with several fields, 
%
%    Output
%
%       model_LLSFDL    - Wx, Wy



   %% optimization parameters
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    rho              = optmParameter.rho;
    theta_x          = optmParameter.thetax;
    theta_y          = optmParameter.thetay;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    num_dim = size(X,2);
    num_class = size(Y,2);
    
    XTX = X'*X;
    R     = pdist2( Y' + eps, Y' + eps, 'cosine' );
    XTY = X'*Y;
    W_x   = (XTX + rho*eye(num_dim)) \ (XTY);
    W_x_1 = W_x;
    if nargin <4
        YTY = Y'*Y;   
        W_y   = (YTY + rho*eye(num_class)) \ (YTY);
        W_y_1 = W_y;
        Lip = sqrt( 3*norm(theta_x^2*XTX)^2 + 3*norm(alpha*R)^2 + 3*norm(theta_x*theta_y*XTY')^2 + 2*norm(theta_y^2*YTY)^2);
    else
        YTY = YP'*YP;  
        XTY = X'*YP;
        W_y   = (YTY + rho*eye(num_class)) \ (YP'*Y);
        W_y_1 = W_y;

        Lip = sqrt( 3*norm(theta_x^2*XTX)^2 + 3*norm(alpha*R)^2 + 3*norm(theta_x*theta_y*XTY')^2 + 2*norm(theta_y^2*YTY)^2);
    end
    iter = 1; oldloss = 0;
    bk = 1; bk_1 = 1; 

    while iter <= maxIter
       % W_x
       W_x_k  = W_x + (bk_1 - 1)/bk * (W_x - W_x_1);
       if nargin <4
           Gw_x_k = W_x_k - 1/Lip * ( theta_x^2* XTX*W_x_k  + theta_x*theta_y*XTY*W_y + alpha * W_x_k*R - theta_x*XTY);
       else
           Gw_x_k = W_x_k - 1/Lip * ( theta_x^2* XTX*W_x_k  + theta_x*theta_y*XTY*W_y + alpha * W_x_k*R - theta_x*X'*Y);
       end
       W_x_1  = W_x;
       W_x    = softthres(Gw_x_k,beta/Lip);
       
       % W_y
       W_y_k  = W_y + (bk_1 - 1)/bk * (W_y - W_y_1);
       if nargin <4
           Gw_y_k = W_y_k - 1/Lip * ( theta_y^2*YTY*W_y_k - theta_y*YTY + theta_x*theta_y*XTY'*W_x);
       else
           Gw_y_k = W_y_k - 1/Lip * ( theta_y^2*YTY*W_y_k - theta_y*YP'*Y + theta_x*theta_y*XTY'*W_x);
       end
       W_y_1  = W_y;
       W_y    = softthres(Gw_y_k,gamma/Lip);
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
      %%
       if nargin <4
           L = (theta_x*X*W_x + theta_x*Y*W_y - Y);
       else
           L = (theta_x*X*W_x + theta_x*YP*W_y - Y);
       end
       DiscriminantLoss = trace(L'* L);
       CorrelationLoss  = trace(R*W_x'*W_x);
       sparesW_x    = sum(sum(W_x~=0));
       sparesW_y    = sum(sum(W_y~=0));
       totalloss = DiscriminantLoss + alpha*CorrelationLoss + beta*sparesW_x + gamma*sparesW_y;

       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    
    model_LLSFDL.W_x = W_x;
    model_LLSFDL.W_y = W_y;
    model_LLSFDL.optmParameter = optmParameter;
end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end
