
function [model_LLSF] = LLSF( X, Y, optmParameter)
% This function is designed to learn label-specific features for each label and
% common features which shared by each combination of two labels.
% 
%    Syntax
%
%       [model_LLSF] = LLSF( X, Y, optmParameter)
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmParameter   - the optimization parameters for LLSF, a struct variable with several fields, 
%   Output
%
%       model_LLSF  - a d by l Coefficient matrix
%
%[1] J. Huang, G.-R Li, Q.-M. Huang and X.-D. Wu. Learning Label Specific Features for Multi-Label Classifcation. 
%    In: Proceedings of the International Conference on Data Mining, 2015.

    
   %% optimization parameters
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    rho              = optmParameter.rho;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    
    W_s   = (XTX + rho*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );
    
    iter  = 1;
    oldloss = 0;
    bk = 1; bk_1 = 1; 
    
    Lip = sqrt(2*(norm(XTX)^2 + norm(alpha*R)^2));
    while iter <= maxIter
       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * gradient(XTX,XTY,alpha,W_s_k,R);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;

       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip);
      
       specificloss = trace((X*W_s - Y)'*(X*W_s - Y));
       traceW_S     = trace(R*W_s'*W_s);
       sparesW_s    = sum(sum(W_s~=0));
       totalloss = specificloss + beta*sparesW_s + alpha*traceW_S; 
       
       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
    end
    
    model_LLSF = W_s;
end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);
end

function gradientvalue = gradient(XTX,XTY,alpha,W,R)
    gradientvalue = (XTX*W - XTY) + alpha * W*R;
end
