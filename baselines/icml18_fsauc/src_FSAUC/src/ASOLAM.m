function [v_bar_s, AUCs, itrs, timing] = ASOLAM(X, y, X_test, y_test, ...
    R, K, gamma)
% Algorithm 1 (SOLAM): not finished yet
% November, 2017
% xzhang131@uiowa.edu
% version 3.0 
% Last updated: January 15, 2018

tic;
timing = [];
itrs = [];
AUCs = [];
v_bar_s = []; 
[d,n] = size(X);
kappa = max(max(X)); % kappa = max(max(X));
m = floor(0.5*log2(2*n/log2(n)))-1;
n0 = floor(n/m);
A = zeros(1,d+2); A_p = zeros(1,d); n_p = 0; A_n = zeros(1,d); n_n = 0; 
v_bar = zeros(d+2,1);

Rk = 2*sqrt(1+2*kappa^2)*R; % R0: updated
p_hat = 0;

sample = randsample(n,n,'false');
for k = 1:m  
    ran = sample((k-1)*n0+1:k*n0);  
    [v_bar, p_hat, A, A_p, A_n, n_p, n_n] = SOLAM1(v_bar, X(:,ran), y(ran,1), ...
        R, gamma, Rk, K, p_hat, A, A_p, A_n, n_p, n_n, k==1); 
    
    % update
    Rk = Rk/2;
    gamma = gamma/2; 
    
    % calculate and display AUC 
    [~,~,~,AUC] = perfcurve(y_test,v_bar(1:d)'*X_test,1);
    disp(['***epoch=', num2str(k), ', itr=', num2str(k*n0), ', AUC=', num2str(AUC)]);
    
    % save epoch level results
    AUCs = cat(1, AUCs, AUC);
    v_bar_s = cat(2, v_bar_s, v_bar);
    itrs = cat(1, itrs, k*n0);
    timing = cat(1, timing, toc); 
end

end