%% compare different methods
clc
clear all

% add the useful path
addpath(genpath('.\plotop'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% some global parameters
%% global pass of all the data
%global gPass
gPass = 15;

%% sequence number
%global gSeqNum
gSeqNum = 1;

%% number of iteration for each sequence
%global gIters
gIters = 5;

%% number of K
%global gCV
gCV = 5;

%% global dimension of the data
%global gData;

% %% a9a
gData(1).datName = 'a9a';
gData(1).datDim = 123;
gData(1).datNum = 32561;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% parameters for each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
optSOLAM = cell(1, gSeqNum);
resSOLAM = cell(1, gSeqNum);

for i = 1:gSeqNum
    resSOLAM{i}.AUC = zeros(gIters, gCV);
    resSOLAM{i}.meanAUC = 0;
    resSOLAM{i}.stdAUC = 0;
    resSOLAM{i}.RT = zeros(gIters, gCV);
    resSOLAM{i}.meanRT = 0;
end

% if matlabpool('size') ~= 0
%     matlabpool close;
% end
% matlabpool(2);

%if exist('par_object')
%    if ~isempty(par_object)
%        delete(par_object);
%    end
%end
%par_object=parpool('local',17);

for i = 1:gSeqNum
    %for i = 1: gSeqNum
    %% get the path of the data
    %datPath = ['..\Data\', gData(i).datName, '\', gData(i).datName];
    datPath = gData(i).datName;
    %% load the data
    [orgFeat, orgLabel] = fnDatLoad(datPath, 1, gData(i).datNum, gData(i).datDim);
    
    fprintf('Successful loading %d-th sequence!\n',i);
    
    %% pass the labels.
    ppLabel = orgLabel;
    %% check the categories of the data, if it is the multi-class data, process it to binary-class
    uLab = unique(orgLabel);
    uNum = length(uLab);
    if uNum > 2
        uSort = randperm(uNum);
        %% positive class
        pIdx = [];
        for k = 1:floor(uNum/2)
            tI = find(orgLabel == uLab(uSort(1, k), 1));
            pIdx = [pIdx, tI'];
        end
        %% negative class
        nIdx = [];
        for k = (uNum - floor(uNum/2)):uNum
            tI = find(orgLabel == uLab(uSort(1, k), 1));
            nIdx = [nIdx, tI'];
        end
        
        %%post-processing
        ppLabel(pIdx, 1) = 1;
        ppLabel(nIdx, 1) = -1;
    end
    
    %% post-processing the data
    ppFeat = zeros(gData(i).datNum, gData(i).datDim);
    for k = 1:gData(i).datNum
        tDat = full(orgFeat(k,:));
        %tDat = tDat - mean(tDat);
        if (norm(tDat) > 0)
            tDat = tDat / norm(tDat);
        end
        ppFeat(k, :) = tDat;
    end
    
    
    %% set the results to zeros
    %1. SOLAM algorithm
    resSOLAM{i}.AUC = zeros(gIters, gCV);
    resSOLAM{i}.meanAUC = 0;
    resSOLAM{i}.stdAUC = 0;
    resSOLAM{i}.RT = zeros(gIters, gCV);
    resSOLAM{i}.meanRT = 0;

    for m = 1:gIters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% generate the training and testing sets using cross validation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vIndices = crossvalind('Kfold', gData(i).datNum, gCV);
        
        for j = 1:gCV
            
            %% get the training samples
            datTrain = ppFeat(vIndices~=j, :);
            labTrain = ppLabel(vIndices~=j);
            
            %% get the testing samples
            datTest = ppFeat(vIndices==j, :);
            labTest = ppLabel(vIndices==j);
            
            %% get the best parameters for each algorithm
            if ((m==1)&&(j==1))
                %1. SOLAM
                 %option = fnEP_CV_SOLAM(datTrain, labTrain, gPass, gCV);
                 %optSOLAM{i} = option;
                 optSOLAM{i}.sC = 100;
                 optSOLAM{i}.sR = 10;
                 optSOLAM{i}.nPass = gPass;
            end
            
            % get the order of training data
            ID=1:size(labTrain, 1);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %1. SOLAM algorithm
            %% number of pass going through the data
            [auc, rt] = fnEP_SOLAM(datTrain, labTrain, datTest, labTest, optSOLAM{i}, ID);
            resSOLAM{i}.AUC(m, j) = auc;
            resSOLAM{i}.RT(m, j) = rt;
            ft = (m-1)*gCV + j;
            fprintf(['Finish SOLAM the %d-th Turn of Seq:', gData(i).datName, '!\n'], ft);          
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% calculate the results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1. SOLAM
    oaAUC = resSOLAM{i}.AUC(:);
    resSOLAM{i}.meanAUC = mean(oaAUC);
    resSOLAM{i}.stdAUC = std(oaAUC);
    
    oaRT = resSOLAM{i}.RT(:);
    resSOLAM{i}.meanRT = mean(oaRT);
    
    %% save results
    %1. SOLAM
    %sPath = ['..\Data\EP_', gData(i).datName, '_SOLAM.mat'];
    sPath = ['EP_', gData(i).datName, '_SOLAM.mat'];
    resData = resSOLAM{i};
    fnSave(sPath, resData);    
end



