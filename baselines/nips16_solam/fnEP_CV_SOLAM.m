%iteration function
function optBest = fnEP_CV_SOLAM(X_train, Y_train, nPass, nCV)
% Used for parameter selecting
% SOLAM: Stochastic Online AUC Maximization
%--------------------------------------------------------------------------
% Input:
%        X_train:    the training instances
%        Y_train:    the vector of lables for X_train
% Output:
%        optBest:    best option
%--------------------------------------------------------------------------

% maximal value
maxAUC = 0;

%% generate the cross idx
CVIdx = crossvalind('Kfold', size(X_train, 1), nCV);

% iterate the variable of eta
for m = 1:9:100
    sC = m;
    for n = -1:5
        sR = 10^n;
            
        curAUC = zeros(1, nCV);
        
        for k = 1:nCV
            
            X_tr = X_train(CVIdx~=k,:);
            Y_tr = Y_train(CVIdx~=k,:);
            X_te = X_train(CVIdx==k,:);
            Y_te = Y_train(CVIdx==k,:);
            
            % dimension of the data.
            nDim = size(X_tr, 2);
            
            nP0_ = 0;
            nV0_ = zeros(nDim + 2, 1);
            nAp0_ = 0;
            nGa0_ = 0;
            
            nV0 = zeros(nDim + 2, 1);
            nV0(1:nDim, 1) = zeros(nDim, 1) + sqrt(sR*sR/nDim);
            nV0(nDim + 1, 1) = sR;
            nV0(nDim + 2, 1) = sR;
            
            nAp0 = 2*sR;
            
            % iteration time.
            nT = 1;
            % pass time.
            nCnt = 1;
            
            while (1)
                if (nCnt > nPass)
                    break;
                end
                
                for t = 1:size(X_tr, 1)
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    tFeat = X_tr(t, :)';
                    tL = Y_tr(t);
                    % get the step parameter
                    nGa = sC / sqrt(nT);
                    
                    % two cases for either postive and negative samples.
                    if tL == 1
                        %step 2
                        nP1_ = ((nT - 1)*nP0_ + 1) / nT;
                        %
                        vWt = nV0(1:nDim, 1);
                        nA = nV0(nDim + 1, 1);
                        vPdv = [2*(1 - nP1_)*(vWt'*tFeat - nA)*tFeat - ...
                            2*(1 + nAp0)*(1 - nP1_)*tFeat;...
                            -2*(1 - nP1_)*(vWt'*tFeat - nA); 0];
                        vPdv = nV0 - nGa*vPdv;
                        
                        vPda = -2*(1 - nP1_)*vWt'*tFeat - ...
                            2*nP1_*(1 - nP1_)*nAp0;
                        vPda = nAp0 + nGa*vPda;
                    else
                        %step 2
                        nP1_ = (nT - 1)*nP0_ / nT;
                        vWt = nV0(1:nDim, 1);
                        nB = nV0(nDim + 2, 1);
                        
                        vPdv = [2*nP1_*(vWt'*tFeat - nB)*tFeat + ...
                            2*(1 + nAp0)*nP1_*tFeat; 0;...
                            -2*nP1_*(vWt'*tFeat - nB)];
                        vPdv = nV0 - nGa*vPdv;
                        
                        vPda = 2*nP1_*vWt'*tFeat - ...
                            2*nP1_*(1 - nP1_)*nAp0;
                        vPda = nAp0 + nGa*vPda;
                    end
                    
                    %normalization.
                    nRv = norm(vPdv(1:nDim,1));
                    if nRv > sR
                        vPdv(1:nDim, 1) = vPdv(1:nDim, 1) / nRv * sR;
                    end
                    if vPdv(nDim + 1, 1) > sR
                        vPdv(nDim + 1, 1) = sR;
                    end
                    if vPdv(nDim + 2, 1) > sR
                        vPdv(nDim + 2, 1) = sR;
                    end
                    
                    nV1 = vPdv;
                    
                    nRa = norm(vPda);
                    if nRa > 2*sR
                        nAp1 = vPda / nRa*(2*sR);
                    else
                        nAp1 = vPda;
                    end
                    
                    %update gamma_
                    nGa1_ = nGa0_ + nGa;
                    %update v_
                    nV1_ = (nGa0_ * nV0_ + nGa * nV0) / nGa1_;
                    
                    
                    %update alpha_
                    nAp1_ = (nGa0_*nAp0_ + nGa*nAp0) / nGa1_;
                    
                    % update the information.
                    nP0_ = nP1_;
                    nV0_ = nV1_;
                    nAp0_ = nAp1_;
                    nGa0_ = nGa1_;
                    
                    nV0 = nV1;
                    nAp0 = nAp1;
                    
                    %update counts
                    nT = nT + 1;
                    
                end
                
                nCnt = nCnt + 1;
            end
            % evaluate the method
            [auc, ~, ~] = fnEvaluate(X_te, Y_te, nV1_(1:nDim, 1));
            curAUC(1, k) = auc;
        end
        
        meanAUC = mean(curAUC);
        if meanAUC > maxAUC
            maxAUC = meanAUC;
            optBest.sC = m;
            optBest.sR = 10^n;
        end
    end
end

optBest.nPass = nPass;












