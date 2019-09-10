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
nPass = 15;
options.nPass = nPass;

%% sequence number
global gSeqNum
gSeqNum = 3;

%% number of K
global gCV
gCV = 5;
options.gCV = gCV;

options.C = 1;

%% global dimension of the data
global gData;

gData(1).datName = 'diabetes';
gData(1).datDim = 8;
gData(1).datNum = 768;
gData(1).datEvaStep = 50;


%% fourclass
gData(2).datName = 'fourclass';
gData(2).datDim = 2;
gData(2).datNum = 862;
gData(2).datEvaStep = 50;

%% german
gData(3).datName = 'german';
gData(3).datDim = 24;
gData(3).datNum = 1000;
gData(3).datEvaStep = 50;

gData(4).datName = 'splice';
gData(4).datDim = 60;
gData(4).datNum = 3175;
gData(4).datEvaStep = 100;

for i = 1:gSeqNum
    
    fig = figure;
    hold on;
    
    %% get the path of the data
    if isunix
        datPath = ['/home/mnatole/MATLAB/datasets/', gData(i).datName, '/', gData(i).datName];
    end
    
    if ismac
        datPath = ['/Users/mnatolejr/Documents/MATLAB/DataSets/', gData(i).datName, '/', gData(i).datName];
    end
    
    if ispc
        datPath = ['C:\Users\mn572395\Documents\MATLAB\DataSets\', gData(i).datName, '\', gData(i).datName];
    end
    
    %% load the data
    [orgFeat, orgLabel] = fnDatLoad(datPath, 1, gData(i).datNum, gData(i).datDim);
    fprintf('Data loaded! \n')
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
    % Normalize the Data
    %ppFeat = (ppFeat - mean(ppFeat)) / std(ppFeat);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% generate the training and testing sets using cross validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% get the training samples
    datTrain = ppFeat;
    labTrain = ppLabel;
    
        
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SPAM Setup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    optL2 = fnEP_CV_SPAM_L2(datTrain,labTrain,options);
    %optNET = fnEP_CV_SPAM_NET(datTrain,labTrain,options);
    options.nStep = gData(i).datEvaStep;
    
    %% Run SPAM_L2
    [TMs, AUCs, ITs] = fnCP_SPAM_L2(datTrain,labTrain,options,optL2);
    fprintf('Finish SPAM - L2!\n');
    
    
    % Plot the Data
    plot(data.TMs,data.AUCs,'LineWidth',3)
    
    %% Run SPAM_NET
    %[TMs, AUCs, ITs] = fnCP_SPAM_NET(datTrain,labTrain,options,optNET);
    %fprintf('Finish SPAM - NET!\n');
    
    
    % Plot the Data
    plot(data.TMs,data.AUCs,'LineWidth',3)   
    
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
    optOPAUC.nEta = 2^(-8);
    optOPAUC.nLambda = optL2.beta;
    optOPAUC.nTau = 50;
    optOPAUC.nPass = nPass;
    optOPAUC.nStep = gData(i).datEvaStep;
    ID = 1:size(datTrain,1);
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
    legend('SPAM - L2','SPAM - Elastic Net', 'SOLAM', 'OPAUC')
    legend('Location','southeast')
    fileName = [gData(i).datName,'_spam.fig'];
    saveas(fig, fileName)
    
end




















