%% compare different methods
clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% global pass of the data
global nPass
nPass = 15;
options.nPass = nPass;

global C;
C = 100;
options.C = C;

global nFeat;
nFeat = 256;

%% sequence number
global gSeqNum
gSeqNum = 3;

%% number of iteration for each sequence
global gIters
gIters = 4;

%% number of K
global gCV
gCV = 5;
options.gCV = gCV;

%% global dimension of the data
global gData;

%% diabetes
gData(1).datName = 'diabetes';
gData(1).datDim = 8;
gData(1).datNum = 768;

%% fourclass
gData(2).datName = 'fourclass';
gData(2).datDim = 2;
gData(2).datNum = 862;

%% german
gData(3).datName = 'german';
gData(3).datDim = 24;
gData(3).datNum = 1000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% parameters for each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resSPAM_L2 = cell(1, gSeqNum);
resSPAM_NET = cell(1, gSeqNum);
resSOLAM = cell(1, gSeqNum);

for i = 1:gSeqNum
    resSPAM_L2{i}.AUC = zeros(gIters, gCV);
    resSPAM_L2{i}.meanAUC = 0;
    resSPAM_L2{i}.stdAUC = 0;
    resSPAM_L2{i}.RT = zeros(gIters, gCV);
    resSPAM_L2{i}.meanRT = 0;
    resSPAM_L2{i}.beta = 0;
    resSPAM_L2{i}.C_L2 = 0;
end

for i = 1:gSeqNum
    resSPAM_NET{i}.AUC = zeros(gIters, gCV);
    resSPAM_NET{i}.meanAUC = 0;
    resSPAM_NET{i}.stdAUC = 0;
    resSPAM_NET{i}.RT = zeros(gIters, gCV);
    resSPAM_NET{i}.meanRT = 0;
    resSPAM_NET{i}.beta1 = 0;
    resSPAM_NET{i}.beta2 = 0;
    resSPAM_NET{i}.C_NET = 0;
end

for i = 1:gSeqNum
    resSOLAM{i}.AUC = zeros(gIters, gCV);
    resSOLAM{i}.meanAUC = 0;
    resSOLAM{i}.stdAUC = 0;
    resSOLAM{i}.RT = zeros(gIters, gCV);
    resSOLAM{i}.meanRT = 0;
end

for i = 1:gSeqNum
    
    % get the path of the data
    datPath = '/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/processed_usps.txt';
    % load the data
    [orgFeat, orgLabel] = fnDatLoad(datPath, 1, 9298, gData(i).datDim);
    
    fprintf('Successfully loaded the data.\n');
    
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
    
    %[ppFeat,xtest,meanxapp,stdxapp] = normalizemeanstd(ppFeat,ppFeat);
    
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
                % Cross Validation for optimal lambda
                optL2 = fnEP_CV_SPAM_L2(datTrain, labTrain,options);
                resSPAM_L2{i}.beta = optL2.beta;
                resSPAM_L2{i}.C_L2 = C;
                fprintf(['Finish SPAM_L2 training: ', gData(i).datName, '!\n']);
                
                optNET = fnEP_CV_SPAM_NET(datTrain, labTrain,options);
                resSPAM_NET{i}.beta1 = optNET.beta1;
                resSPAM_NET{i}.beta2 = optNET.beta2;
                resSPAM_NET{i}.C_NET = C;
                fprintf(['Finish SPAM_NET training: ', gData(i).datName, '!\n']);
                
                optSOLAM = fnEP_CV_regSOLAM(datTrain,labTrain, options);
                resSOLAM{i}.optBest = optSOLAM;
                fprintf(['Finish SOLAM training: ', gData(i).datName, '!\n']);
                
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % SPAM algorithm
            
            % 1. SPAM_L2
            [auc, rt] = fnEP_SPAM_L2(datTrain, labTrain, datTest, labTest,options,optL2);
            resSPAM_L2{i}.AUC(m, j) = auc;
            resSPAM_L2{i}.RT(m, j) = rt;
            ft = (m-1)*gCV + j;
            fprintf(['Finish SPAM_L2 the %d-th Turn of Seq:', gData(i).datName, '!\n'], ft);
            
            % 2. SPAM_NET
            [auc, rt] = fnEP_SPAM_NET(datTrain, labTrain, datTest, labTest,options,optNET);
            resSPAM_NET{i}.AUC(m, j) = auc;
            resSPAM_NET{i}.RT(m, j) = rt;
            ft = (m-1)*gCV + j;
            fprintf(['Finish SPAM_NET the %d-th Turn of Seq:', gData(i).datName, '!\n'], ft);
            
            %3. SOLAM
            [auc, rt] = fnEP_regSOLAM(datTrain,labTrain,datTest,labTest,options,optSOLAM);
            resSOLAM{i}.AUC(m, j) = auc;
            resSOLAM{i}.RT(m, j) = rt;
            ft = (m-1)*gCV + j;
            fprintf(['Finish SOLAM the %d-th Turn of Seq:', gData(i).datName, '!\n'], ft);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% calculate the results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. SPAM_L2
    oaAUC = resSPAM_L2{i}.AUC(:);
    resSPAM_L2{i}.meanAUC = mean(oaAUC);
    resSPAM_L2{i}.stdAUC = std(oaAUC);
    
    oaRT = resSPAM_L2{i}.RT(:);
    resSPAM_L2{i}.meanRT = mean(oaRT);
    
    %% save results
    if ispc
        sPath = ['..\SPAM\Results\EP_', gData(i).datName, '_SPAM_L2.mat'];
    end
    
    if isunix
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_L2.mat'];
    end
    
    if ismac
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_L2.mat'];
    end
    
    resData = resSPAM_L2{i};
    fnSave(sPath, resData);
    resData
    
    % 2. SPAM_NET
    oaAUC = resSPAM_NET{i}.AUC(:);
    resSPAM_NET{i}.meanAUC = mean(oaAUC);
    resSPAM_NET{i}.stdAUC = std(oaAUC);
    
    oaRT = resSPAM_NET{i}.RT(:);
    resSPAM_NET{i}.meanRT = mean(oaRT);
    
    %% save results
    if ispc
        sPath = ['..\SPAM\Results\EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    if isunix
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    if ismac
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    resData = resSPAM_NET{i};
    fnSave(sPath, resData);
    resData
    
    % 3. SOLAM
    oaAUC = resSOLAM{i}.AUC(:);
    resSOLAM{i}.meanAUC = mean(oaAUC);
    resSOLAM{i}.stdAUC = std(oaAUC);
    
    oaRT = resSOLAM{i}.RT(:);
    resSOLAM{i}.meanRT = mean(oaRT);
    
    %% save results
    if ispc
        sPath = ['..\SPAM\Results\EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    if isunix
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    if ismac
        sPath = ['../SPAM/Results/EP_', gData(i).datName, '_SPAM_NET.mat'];
    end
    
    resData = resSOLAM{i};
    fnSave(sPath, resData);
    resData
    %emailMe('Data set has completed processing');
end

WarnWave = [sin(1:.6:400), sin(1:.7:400), sin(1:.4:400)];
Audio = audioplayer(WarnWave, 22050);
play(Audio);





