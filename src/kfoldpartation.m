function partationData = kfoldpartation(dataX, dataY, no_fold, rand_idx)
% KFOLDPARTATION - Create k-fold cross-validation partitions
%
% This function creates k-fold cross-validation partitions for multi-label data.
% Each fold contains training and testing sets with appropriate data splits.
%
% Inputs:
%   dataX - Feature matrix (n_samples x n_features)
%   dataY - Label matrix (n_samples x n_labels)
%   no_fold - Number of folds for cross-validation
%   rand_idx - Random permutation indices for data shuffling
%
% Outputs:
%   partationData - Cell array containing fold data structures
%                   Each cell contains fields: X, Y (training), Xt, Yt (testing)
%
% Example:
%   rand_idx = randperm(size(dataY,1));
%   folds = kfoldpartation(dataX, dataY, 5, rand_idx);

% Validate inputs
if nargin < 4
    error('kfoldpartation:InvalidInput', 'All four arguments are required');
end
if no_fold < 2
    error('kfoldpartation:InvalidFolds', 'Number of folds must be at least 2');
end
if length(rand_idx) ~= size(dataY, 1)
    error('kfoldpartation:InvalidIndex', 'Random index length must match number of samples');
end
partationData = cell(1,no_fold);
N          = size(dataY,1);
batchsize  =ceil(N/no_fold);
Idx        =1:N;
for foldNo=1:no_fold
    testStartIdx=batchsize*(foldNo-1)+1;
    testEndIdx=min((batchsize*foldNo),N);
    
    trainIdx = setdiff(Idx,testStartIdx:testEndIdx);
    
    partationData{foldNo}.X = dataX(rand_idx(trainIdx),:);
    partationData{foldNo}.Y = dataY(rand_idx(trainIdx),:);
    
    partationData{foldNo}.Xt = dataX(rand_idx(testStartIdx:testEndIdx),:);
    partationData{foldNo}.Yt = dataY(rand_idx(testStartIdx:testEndIdx),:);
    
end
end
