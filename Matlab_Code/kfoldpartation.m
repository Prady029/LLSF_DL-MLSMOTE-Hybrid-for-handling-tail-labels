function  partationData    = kfoldpartation( dataX, dataY, no_fold, rand_idx)
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
