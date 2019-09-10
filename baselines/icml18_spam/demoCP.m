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
global nPass
nPass = 25;
options.nPass = nPass;

%% sequence number
global gSeqNum
gSeqNum = 4;

%% number of K
global gCV
gCV = 5;
options.gCV = gCV;

%% global dimension of the data
global gData;

%% splice
gData(1).datName = 'splice';
gData(1).datDim = 60;
gData(1).datNum = 3175;
gData(1).datEvaStep = 100;

 %% usps
gData(2).datName = 'usps';
gData(2).datDim = 256;
gData(2).datNum = 9298;
gData(2).datEvaStep = 1000;

%% a9a data
gData(3).datName = 'a9a';
gData(3).datDim = 123;
gData(3).datNum = 32561;
gData(3).datEvaStep = 5000;

%% sector
gData(4).datName = 'sector';
gData(4).datDim = 55197;
gData(4).datNum = 9619;
gData(4).datEvaStep = 2000;

gData(5).datName = 'diabetes';
gData(5).datDim = 8;
gData(5).datNum = 768;
gData(5).datEvaStep = 50;

gData(6).datName = 'german';
gData(6).datDim = 24;
gData(6).datNum = 1000;
gData(6).datEvaStep = 50;

for i = 1:gSeqNum
    
    fig = figure;
    hold on;
    
    %% get the path of the data
    if isunix
        datPath = ['/home/mnatole/MATLAB/datasets/', gData(i).datName, '/', gData(i).datName];
    end
    
    if ismac
        datPath = ['/Users/MNATOLE/MATLAB/DataSets/', gData(i).datName, '/', gData(i).datName];
    end
    
    if ispc
        datPath = ['C:\Users\mn572395\Documents\MATLAB\DataSets\', gData(i).datName, '\', gData(i).datName];
    end
    
    %% load the data
    [orgFeat, orgLabel] = fnDatLoad(datPath, 1, gData(i).datNum, gData(i).datDim);
    fprintf('Data has been loaded!\n');
    
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
    
    %% INITIAL SETUP
    % need to find the indices of positive and negative examples
    index_pos = find(ppLabel == 1);
    index_neg = find(ppLabel == -1);
    
    % need to find the number of samples for each class and in total
    n_pos = length(index_pos);
    n_neg = length(index_neg);
    
    % Determine p
    options.p = n_pos / (n_pos + n_neg);
    
    % Need to determine b,m_pos, and m_neg
    options.m_pos = mean(ppFeat(index_pos,:));
    options.m_neg = mean(ppFeat(index_neg,:));
        
    %% get the training samples
    datTrain = ppFeat;
    labTrain = ppLabel;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SPAM Setup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %options.param = fnEP_CV_SPAM_L2(datTrain,labTrain,options);
    %options.param2 = fnEP_CV_SPAM_NET(datTrain,labTrain,options);
    options.param.beta = 0.01;
    options.param2.beta1 = 0.01;
    options.param2.beta2 = 0;
    options.nStep = gData(i).datEvaStep;
    
    %% Run SPAM_L2
    [TMs, AUCs, ITs] = fnCP_SPAM_L2(datTrain,labTrain,options);
    fprintf('Finish SPAM - L2!\n');
    
    % Get Data
    data.ITs = ITs;
    data.TMs = TMs;
    data.AUCs = AUCs;
    
    % Plot the Data
    plot(data.TMs,data.AUCs,'LineWidth',3)
    
    %% Run SPAM_NET
%     [TMs, AUCs, ITs] = fnCP_SPAM_NET(datTrain,labTrain,options,ID);
%     fprintf('Finish SPAM - NET!\n');
%     
%     % Get Data
%     data.ITs = ITs;
%     data.TMs = TMs;
%     data.AUCs = AUCs;
%     
%     % Plot the Data
%     plot(data.TMs,data.AUCs,'LineWidth',3)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SOLAM
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    optSOLAM.lambda = optL2.beta;
    optSOLAM.sC = optL2.C;
    
    [TMs, AUCs, ITs] = fnCP_regSOLAM(datTrain, labTrain, options,optSOLAM);
    
    % Get Data
    data.ITs = ITs;
    data.TMs = TMs;
    data.AUCs = AUCs;
    
    % Plot the Data
    plot(data.TMs,data.AUCs,'LineWidth',3)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% One-Pass algorithm (ICML 2013)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ID = 1:size(datTrain,1);
    optOPAUC.nEta = 2^(-8);
    optOPAUC.nLambda = optL2.beta;
    optOPAUC.nTau = 50;
    optOPAUC.nPass = nPass;
    optOPAUC.nStep = gData(i).datEvaStep;
    
    [AUCs, ITs, TMs] = fnCP_OPAUC(datTrain, labTrain, optOPAUC, ID);
    fprintf('Finish OPAUC!\n');
    
    data.ITs = ITs;
    data.TMs = TMs;
    data.AUCs = AUCs;
    
    plot(data.TMs,data.AUCs,'LineWidth',3)
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Finish the Plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    title(gData(i).datName);
    ylabel('AUC')
    xlabel('Time')
    legend('SPAM - L2','SOLAM','OPAUC')
    legend('Location','southeast')
    fileName = [gData(i).datName,'_spam_All.fig'];
    saveas(fig, fileName)
    
    %emailMe('Plot completed')
end