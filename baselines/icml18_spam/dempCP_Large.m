%% compare different methods
clc
clear all

% add the useful path
if isunix
    addpath(genpath('./plotop'));
end

if ismac
    addpath(genpath('./plotop'));
end

if ispc
    addpath(genpath('.\plotop'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% some global parameters
%% global pass of all the data
global epochs
epochs = 40;

%% sequence number
global gSeqNum
gSeqNum = 4;

%% number of K
global gCV
gCV = 5;

%% global dimension of the data
global gData;

%% splice
gData(1).datName = 'splice';
gData(1).datDim = 60;
gData(1).datNum = 3175;
gData(1).lambda = 1e-3;
gData(1).datEvaStep = 100;

 %% usps
gData(2).datName = 'usps';
gData(2).datDim = 256;
gData(2).datNum = 9298;
gData(2).lambda = 1e-3;
gData(2).datEvaStep = 1000;

%% a9a data
gData(3).datName = 'a9a';
gData(3).datDim = 123;
gData(3).datNum = 32561;
gData(3).lambda = 1e-3;
gData(3).datEvaStep = 5000;

%% acoustic
gData(4).datName = 'sector';
gData(4).datDim = 55197;
gData(4).datNum = 9619;
gData(4).lambda = 1e-3;
gData(4).datEvaStep = 2000;

for i = 1:gSeqNum
    
    if i ~= 2
        continue
    end
    fig = figure;
    hold on;
    
    batchSize = ceil(0.10*gData(i).datNum);
    
    %% get the path of the data
    if isunix
        datPath = '/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/usps';
    end
    
    if ismac
        datPath = ['/Users/MNATOLE/MATLAB/DataSets/', gData(i).datName, '/', gData(i).datName];
    end
    
    if ispc
        datPath = ['C:\Users\mn572395\Documents\MATLAB\DataSets\', gData(i).datName, '\', gData(i).datName];
    end
    
    %% load the data
    [orgFeat, orgLabel] = fnDatLoad(datPath, 1, gData(i).datNum, gData(i).datDim);
    
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
        tDat = tDat - mean(tDat);
        if (norm(tDat) > 0)
            tDat = tDat / norm(tDat);
        end
        ppFeat(k, :) = tDat;
    end
    
    %% INITIAL SETUP
    % Normalize the Data
    %ppFeat = (ppFeat - mean(ppFeat)) / std(ppFeat);
    
    % need to find the indices of positive and negative examples
    index_pos = find(ppLabel == 1);
    index_neg = find(ppLabel == -1);
    
    % need to find the number of samples for each class and in total
    n_pos = length(index_pos);
    n_neg = length(index_neg);
    
    % Determine p
    p = n_pos / (n_pos + n_neg);
    
    % Need to determine b,m_pos, and m_neg
    optSOLAM.m_pos = (1/n_pos)*sum(ppFeat(index_pos,:));
    optSOLAM.m_neg = (1/n_neg)*sum(ppFeat(index_neg,:));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% generate the training and testing sets using cross validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vIndices = crossvalind('Kfold', gData(i).datNum, gCV);
    
    %% get the training samples
    datTrain = ppFeat(vIndices~=1, :);
    labTrain = ppLabel(vIndices~=1);
    
    %% get the testing samples
    datTest = ppFeat(vIndices==1, :);
    labTest = ppLabel(vIndices==1);
        
   

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Finish the Plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    title(gData(i).datName);
    ylabel('AUC')
    xlabel('Time')
    legend('regSOLAM','expSOLAM1','expSOLAM2')
    
    fileName = [gData(i).datName,'_spd_reg_v1_v2_time.fig'];
    saveas(fig, fileName)
        
end

emailMe('Figures have completed')



















