function [AUC, RT] = fnEP_SPAM_L2(X_train, Y_train, X_test, Y_test, options, optL2)
% SPAM_L2: Stochastic Proximal AUC Maximization with L2 penalty term
%--------------------------------------------------------------------------
% Input:
%        X_train:    the training instances
%        Y_train:    the vector of lables for X_train
%         X_test:    the testing instances
%         Y_test:    the vector of lables for X_test
%        options:    a struct containing nPass
% Output:
%         TMs:  time consumed by this algorithm once
%         AUC:  area under ROC curve
%--------------------------------------------------------------------------

% Setup
beta = optL2.beta;
C = optL2.C;
nPass = options.nPass;

% Dimension of the data
[n, d] =  size(X_train);

% need to find the indices of positive and negative examples
index_pos = find(Y_train == 1);
index_neg = find(Y_train == -1);

% need to find the number of samples for each class and in total
n_pos = length(index_pos);
n_neg = length(index_neg);

% Determine p
p = n_pos / n;

% Need to determine m_pos and m_neg
m_pos = mean(X_train(index_pos,:));
m_neg = mean(X_train(index_neg,:));

% Variables
w = zeros(d,1);
w_bar = zeros(d,1);
nT = 1;

% pass time.
nCnt = 1;

tS = cputime;

while (1)
    if (nCnt > nPass)
        break;
    end
    
    for j = 1:n
        % Get the sample
        tFeat = X_train(j, :)';
        tL = Y_train(j);
        
        % get the step parameter
        eta = C / sqrt(nT);
        
        % Compute a, b, and alpha
        a = m_pos*w;
        b = m_neg*w;
        alpha = b - a;
        
        if tL == 1
            % Determine the derivative
            wPdw = 2*(1 - p)*(w'*tFeat - a)*tFeat - ...
                2*(1 + alpha)*(1 - p)*tFeat;
        end
        
        if tL == -1
            % Determine the derivative
            wPdw = 2*p*(w'*tFeat - b)*tFeat + ...
                2*(1 + alpha)*p*tFeat;
        end
        
        % Gradient Step
        u = w - eta*wPdw;
        
        % Proximal Step
        w = u / (1 + beta*eta);
        
        % Average
        w_bar = (w_bar*(nT-1) + w)/nT;
        
        %update counts
        nT = nT + 1;
        
    end
    
    nCnt = nCnt + 1;
end

% end point
tE = cputime;

% evaluate the method
[AUC, ~, ~] = fnEvaluate(X_test, Y_test, w_bar);

RT = tE - tS;

end


