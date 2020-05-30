function [IR_label, meanir]=Imbalance_ratio(Y)

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