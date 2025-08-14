function [IR_label, meanir]=Imbalance_ratio(Y)
% IMBALANCE_RATIO - Calculate imbalance ratio for each label
%
% This function computes the imbalance ratio for each label in a multi-label
% dataset. The imbalance ratio is defined as the ratio of the maximum number
% of instances per label to the number of instances for each specific label.
%
% Inputs:
%   Y - Binary label matrix (n_samples x n_labels)
%
% Outputs:
%   IR_label - Imbalance ratio for each label (1 x n_labels)
%   meanir - Mean imbalance ratio across all labels (scalar)
%
% Example:
%   Y = [1 0 1; 0 1 0; 1 1 0; 1 0 0];
%   [IR_label, meanir] = Imbalance_ratio(Y);

% Validate input
if nargin < 1
    error('Imbalance_ratio:InvalidInput', 'Label matrix Y is required');
end
if ~islogical(Y) && ~all(ismember(Y(:), [0 1]))
    warning('Imbalance_ratio:NonBinary', 'Y should be a binary matrix');
end

labelWiseIns = sum(Y,1);
IR_label     = max(labelWiseIns)./(eps+labelWiseIns);
meanir        = sum(IR_label)/size(Y,2);

% [numInstances,L]=size(Y); %  L=numLabels Labels in dataset
% IR_label=zeros(1,L);
% %calculate IRLbls and meanIR
% for i=1:L
%     %b(i)=sum(Y(:,i));
%     IR_label(i)=max(sum(Y))/sum(Y(:,i));
% end
% IR_label(isinf(IR_label)|isnan(IR_label))=0;
% meanir=(1/L)*sum(IR_label);
end