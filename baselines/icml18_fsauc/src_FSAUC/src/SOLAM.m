function [itrs, v_bar_s, AUCs, timing] ...
    = SOLAM( v, X, y, X_test, y_test, zeta, R, n_delta, l1)
% SOLAM Algorithm
% November 1, 2017
% xzhang131@uiowa.edu
% version 3.0  

tic;

if nargin <= 8
    l1 = 0; % default: using l1 norm
end

% returns
itrs = [];
AUCs = [];
v_bar_s = [];  
timing = [];

% initialization
[d,n] = size(X);
kappa = max(max(X));
delta = floor(n/n_delta);  % store n_delta intermediate results
alpha = 0; 
p_hat = 0;    
alpha_bar = 0;
gamma_bar = 0;
v_bar = zeros(d+2,1);

% a copy of v
w = v([1:d]); 
a = v(d+1);
b = v(d+2);
w_bar = zeros(d,1);
a_bar = 0; 
b_bar = 0; 

sample = randsample(n,n,'false'); % one pass
for i = 1:n    
    xi = X(:,sample(i));
    yi = y(sample(i));
    gamma = zeta/i^0.5;
    p_hat = ((i-1)*p_hat+(yi==1))/i;
    
    % compute gradient
%     pred = v'*[xi;0;0]; 
    pred = w'*xi; 
    temp = 2*((yi==-1)*p_hat-(yi==1)*(1-p_hat));
%     ga = 2*(yi==1)*(1-p_hat)*(v(d+1)-pred);
    ga = 2*(yi==1)*(1-p_hat)*(a-pred);
%     gb = 2*(yi==-1)*p_hat*(v(d+2)-pred);
    gb = 2*(yi==-1)*p_hat*(b-pred);
    gw = ((1+alpha)*temp-ga-gb)*xi;
    galpha = temp*pred-2*p_hat*(1-p_hat)*alpha;
    
    % update 
%     v = v-gamma*[gw;ga;gb]; %gv = [gw;ga;gb];
    w = w-gamma*gw; 
    a = a-gamma*ga; 
    b = b-gamma*gb;  
    
    alpha = alpha+gamma*galpha;
    
    % projection
    % pro to omega1: w
%     v_norm = norm(v([1:d]), 2);
    if l1
        w = ProjectOntoL1Ball(w, R);
    else
        v_norm = norm(w, 2);
        if v_norm > R
%             v([1:d]) = v([1:d])*R/v_norm;
            w = w*R/v_norm;
        end
    end
%     v(d+1) = sign(v(d+1))*min(abs(v(d+1)),R*kappa); % pro to omega1: a
    a = sign(a)*min(abs(a),R*kappa); 
%     v(d+2) = sign(v(d+2))*min(abs(v(d+2)),R*kappa); % pro to omega1: b
    b = sign(b)*min(abs(b),R*kappa); 
    % pro to omega2: alpha
    a_tmp = min([2*R*kappa,alpha]);
    alpha = max([-2*R*kappa,a_tmp]);
    
    % average
    gamma_bar_old = gamma_bar;
    gamma_bar = gamma_bar + gamma;
%     v = [w;a;b];
%     v_bar = (gamma_bar_old*v_bar+gamma*v)/gamma_bar;
%     v_bar = (gamma_bar_old*v_bar+gamma*[w;a;b])/gamma_bar;
    w_bar = (gamma_bar_old*w_bar+gamma*w)/gamma_bar;
    a_bar = (gamma_bar_old*a_bar+gamma*a)/gamma_bar;
    b_bar = (gamma_bar_old*b_bar+gamma*a)/gamma_bar;
    alpha_bar = (gamma_bar_old*alpha_bar+gamma*alpha)/gamma_bar;
       
    % compute AUC and save results
    if mod(i,delta) == 0      
%         [~,~,~,AUC] = perfcurve(y_test,v_bar(1:d)'*X_test,1);        
        [~,~,~,AUC] = perfcurve(y_test,w_bar(1:d)'*X_test,1);        
        disp([num2str(i/delta), '. AUC=', num2str(AUC)]);
        AUCs = cat(1, AUCs, AUC);
%         v_bar_s = cat(2, v_bar_s, v_bar); 
        v_bar_s = cat(2, v_bar_s, [w_bar;a_bar;b_bar]); 
        itrs = cat(1, itrs, i);       
        timing = cat(1, timing, toc); 
    end
end

end