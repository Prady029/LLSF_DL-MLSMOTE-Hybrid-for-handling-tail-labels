function config = get_config()
% GET_CONFIG - Configuration settings for LLSF-DL MLSMOTE experiments
%
% This function returns a configuration structure containing all the
% hyperparameters and settings used across different experiments.
%
% Outputs:
%   config - Structure containing configuration parameters
%
% Usage:
%   config = get_config();
%   optmParameter = config.optmParameter;

config = struct();

% Dataset configuration
config.datasets = [{'../Datasets/genbase'};
                   {'../Datasets/emotions'};
                   {'../Datasets/rcv1-sample1'};
                   {'../Datasets/recreation'}];

% Experiment parameters
config.ttlFold = 5;              % Total number of folds
config.maxSelfIter = 3;          % Maximum self-learning iterations
config.ttlEva = 15;              % Total evaluation metrics
config.maxIter = 100;            % Maximum iterations for optimization
config.a = 5;                    % Number of synthetic samples per minority instance

% LLSF-DL optimization parameters
config.optmParameter = struct();
config.optmParameter.maxIter = config.maxIter;
config.optmParameter.minimumLossMargin = 0.001;
config.optmParameter.bQuiet = 1;
config.optmParameter.alpha = 4^-3;       % label correlation
config.optmParameter.beta = 4^-2;        % sparsity of label specific features
config.optmParameter.gamma = 4^-1;       % sparsity of label specific dependent labels
config.optmParameter.rho = 0.1;
config.optmParameter.thetax = 1;
config.optmParameter.thetay = 1;

% Dataset-specific parameter recommendations
% Parameters: alpha, beta, gamma, rho
% for emotions      : 4^{5, 3, 3}, 0.1
% for rcv1subset1   : 4^{5, 3, 3}, 1
% for genbase       : 4^{-3,-2,-1}, 0.1
% for recreation    : 4^{6, 4, 5}, 1

end
