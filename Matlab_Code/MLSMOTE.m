function [train_data, train_target]=MLSMOTE(X,L,Y,a)
% for each label in L do
Data=[X Y];
k=5;
NEW_Data=[];
% calculate at which row this label is 1 and take all the instance
% from X.
[r,~]=find(Y(:,L));
minBag=Data(r,:);  % get all instances of label line 7
for j=1:size(minBag,1)   %for each sample in minBag line 8
    distance= pdist2(minBag(j,:),minBag);  % calculate distance
    [distance_sort,Idx]=sort(distance,'ascend');  % sort Smaller To Largest Distance line 10
    %Neighbour set selection
    %neighbour=distance_sort(Idx(:,2:k+1));
    for i=1:a
    if size(Idx,2)>=k+1
        
        neighbour_IDX=Idx(:,2:k+1);
       % neighbour=full(minBag(Idx(:,2:k+1),:));
        neighbour=full(minBag(neighbour_IDX,:));
       refNeigh=neighbour(randi(length(neighbour_IDX)),:);
        
    else
        neighbour_IDX=Idx;
        neighbour=full(minBag(Idx,:));
      
        refNeigh=neighbour(randi(length(neighbour_IDX)),:);
        
    end
    % refNeigh=neighbour(randperm(size(neighbour,1),1),:); %randomly select neighbour
    %feature set and Label set selection
    
    synthSmpl_feature=zeros(1,size(X,2));
    %feature set assignment
    diff=refNeigh(:,1:size(X,2))-minBag(j,1:size(X,2));
    offset=diff*rand(1);
    value=minBag(j,1:size(X,2))+offset;
    synthSmpl_feature(:,1:size(X,2))=value;
    %label set assignment
%             lblCounts=nnz(minBag(j,size(X,2)+1:size(minBag,2)));
%             lblCounts=lblCounts+nnz(neighbour(:,size(X,2)+1:size(minBag,2)));
%             labels=lblCounts>((k+1)/2);
    %label set assignment
    
    lblCounts=minBag(j,size(X,2)+1:size(minBag,2));
    lblCounts=lblCounts+sum(neighbour(:,size(X,2)+1:size(minBag,2)),1);
    labels=lblCounts>((k+1)/2);
    synthSmpl_label=labels;
    synthSmpl(i,:)=[synthSmpl_feature,synthSmpl_label];
    end
    NEW_Data=[NEW_Data;synthSmpl];
end
train_data=NEW_Data(:,1:size(X,2));
train_target=NEW_Data(:,((size(X,2)+1):end));
end

%     %Ranking case
%     for i=1:size(Y,2)
%         c0=0;c1=0;
%         if (minBag(j,(1:size(X,2)+i)) == 1)
%             c1=c1+1;
%         elseif (minBag(j,(1:size(X,2)+i)) == 0)
%             c0=c0+1;
%         end
%         for ii=1:size(neighbour,1)
%             if (neighbour(ii,(size(X,2)+i)) == 1)
%                 c1=c1+1;
%             elseif (neighbour(ii,(size(X,2)+i)) == 0)
%                 c0=c0+1;
%             end
%         end
%         if(c1*2>=(c0+c1))
%             synthSmpl_label(i)=1;
%         else
%             synthSmpl_label(i)=0;
%         end
%     end




%{
            %label set assignment
            lblCounts=nnz(minBag(j,size(X,2)+1:size(minBag,2)));
            lblCounts=lblCounts+nnz(neighbour(:,size(X,2)+1:size(minBag,2)));
            labels=lblCounts>((k+1)/2);
%}