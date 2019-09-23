function [itrs, w_s, AUCs, timing] = OPAUC(X,y,X_test,y_test,eta,lambda,n_delta,u)
% Usage: OPAUC algorithm 
%        nargin < 8: standard OPAUC
%        nargin == 8: OPAUCr (OPAUC for large scale data)
% Author: xzhang
% Date: 2017.11.16
% Version: 1.2

tic; 
'new~'

% Returns
AUCs = [];
itrs = [];
timing = [];
w_s = [];
y = sparse(y);

% parameters
[d,n] = size(X);
if nargin < 8 || ~u 
    init = @zeros;
    u = d;
    std = 1; 
else
    disp('large scale and going with sparse setting ...')
    std = 0; 
    init = @sparse;
    R_hat_p = init(1,u);
    R_hat_n = init(1,u);
end
T_p = 0; 
T_n = 0; 
c_p = init(d,1); c_n = init(d,1);
w = init(d,1);
Gamma_p = init(d,u);
Gamma_n = init(d,u);
delta = floor(n/n_delta);
  
sample = randsample(n, n, 'false');
for i = 1:n
    xi = X(:,sample(i));
    yi = y(sample(i));
    if yi == 1
        T_p = T_p+1;
        c_p = c_p+(xi-c_p)/T_p;
        if std 
            [Gamma_p,g] = std_update(xi,yi,w,lambda,Gamma_p,c_p,T_p,Gamma_n,c_n); 
        else
            [Gamma_p,g,R_hat_p] = large_update(xi,yi,w,lambda,u,Gamma_p,R_hat_p,Gamma_n,c_n,R_hat_n,T_n);
        end
    else
        T_n = T_n+1;
        c_n = c_n+(xi-c_n)/T_n;
        if std
            [Gamma_n,g] = std_update(xi,yi,w,lambda,Gamma_n,c_n,T_n,Gamma_p,c_p);             
        else 
            [Gamma_n,g,R_hat_n] = large_update(xi,yi,w,lambda,u,Gamma_n,R_hat_n,Gamma_p,c_p,R_hat_p,T_p);            
        end
    end
    w = w-eta*g; % update model 
    
    % collect intermediate results
    if mod(i,delta) == 0 
        [~,~,~,AUC] = perfcurve(y_test,w'*X_test,1);
        disp(['***OPAUC: itr=', num2str(i), ', AUC=', num2str(AUC)]);
        AUCs = cat(1, AUCs, AUC); 
        itrs = cat(1, itrs, i);       
        timing = cat(1, timing, toc);
        w_s = cat(2, w_s, w); 
    end
end
end

function [Gamma,g] = std_update(xi,yi,w,lambda,Gamma,c,T,Gamma_op,c_op)
    tmp = c; 
    Gamma = Gamma+(xi*xi'-Gamma)/T+tmp*tmp'-c*c';       
    g = lambda*w-yi*xi+yi*c_op+(xi-c_op)*(xi-c_op)'*w+Gamma_op*w;
end

function [Gamma,g,R_hat] = large_update(xi,yi,w,lambda,u,Gamma,R_hat,Gamma_op,c_op,R_hat_op,T_op)    
    % ri = randn(u,1); 
    ri = sparse(randn(u,1)); 
    R_hat = R_hat + ri'/u; % update R_hat
    Gamma = Gamma + xi*ri'/u; % update Z
    tmp = xi-c_op; 
    g = lambda*w-yi*(tmp)+(tmp)*((tmp)'*w); %
    if T_op > 0
        % c_hat_op = (c_op*R_hat_op)/T_op;
        g = g + Gamma_op*(Gamma_op'*w)/T_op-(c_op*(R_hat_op*R_hat_op')*(c_op'*w))/(T_op^2); 
    end
end