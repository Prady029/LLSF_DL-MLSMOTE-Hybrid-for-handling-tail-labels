% ========================================================================================
% To reproduct the results published in our  paper, the following parameters are suggested.
% On the other hand, best parameter settings could be searched by cross validation on the 
% training data by the function named 'LLSF_DL_adaptive_validate.m'
% ========================================================================================
% 
% LLSF-DL:
% Parameters          : alpha, beta, gamma, rho
%      for cal500     : 4^{5, 4, 3}, 10 
%      for genbase    : 4^{-3,-2,-1}, 0.1
%      for languagelog: 4^{5, 3, 5},1
%      for medical    : 4^{-1,-2,-1}, 0.1
%
%      for arts       : 4^{5, 3, 5},1
%      for recreation : 4^{6, 4, 5},1
%      for education  : 4^{5, 3, 3}, 0.1    
%      for science    : 4^{5, 3, 5},1
%
%      for rcv1subset1: 4^{5, 3, 3}, 1    
%	   for bibtex     : 4^{2, 0, -1}, 0.1      2^10, 2^5, 2^2, 0.1
%      for corel16k001: 4^{8, 6, 1}, 1
%      for corel5k    : 4^{5, 2, 1}, 1
%      for delicious  : 4^{5, 3, -5}, 1     
%
%      for imdb       : 4^{10, 7, -5}, 10;  4^{10, 8, 1}, 0.1;  4^{10, 7, 1}, 10;
%      for bookmark   : 4^{4, 2, 3}, 1   
%      for nuswide    : 4^{10, 7, 5}, 1    2^20, 2^14, 2^-10, 1
% =======================================================================================
function [optmParameter, modelparameter] =  initialization
% for simplicity, parameters (i.e., alpha, beta, and rho) for LLSF are the same as LLSF-DL
    optmParameter.alpha   = 4^5;  % 4.^[-5:5] % label correlation
    optmParameter.beta    = 4^4;  % 4.^[-5:5] % sparsity of feature
    optmParameter.gamma   = 4^3;  % 4.^[-5:5] % sparsity of label , LLSF-DL 
    optmParameter.rho     = 10;
    
    optmParameter.thetax  = 1;
    optmParameter.thetay  = 1;
    
    optmParameter.maxIter           = 100;
    optmParameter.minimumLossMargin = 0.001;

   %% Model Parameters
    modelparameter.crossvalidation    = 1; % {0,1}
    modelparameter.cv_num             = 5;
    modelparameter.L2Norm             = 1; % {0,1}
    modelparameter.tuneparameter      = 0; % {0,1}
    
    modelparameter.searchbestparas    = 0; % {0,1}
    modelparameter.alpha_searchrange  = 4.^[-5:5]; 
    modelparameter.beta_searchrange   = 4.^[-5:5];
    modelparameter.gamma_searchrange  = 4.^[-5:5];
    modelparameter.rho_searchrange    = 10.^[-1:1];

end