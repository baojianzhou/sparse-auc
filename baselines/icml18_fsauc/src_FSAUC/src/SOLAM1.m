function [v_bar, p_hat, A, A_p, A_n, n_p, n_n] = SOLAM1( v1, X, y, R, gamma, R0, K, p_hat, A, A_p, A_n, n_p, n_n, init_ep)
% Algorithm 1 (SOLAM)
% November, 2017
% xzhang131@uiowa.edu
% version 3.0 
% Last updated: January 15, 2018

% common parameters
delta = 0.1;
kappa = get_kappa(X); % kappa = max(max(X));
[d,n] = size(X);

% initial results
alpha1 = A*v1; alpha = alpha1;
% v = v1;
w1 = v1([1:d]); w = w1; 
a1 = v1(d+1); a = a1; 
b1 = v1(d+2); b = b1; 

% averaged results
% v_bar = zeros(d+2,1);
w_bar = zeros(d,1);
a_bar = 0;
b_bar = 0;
alpha_bar = 0;
gamma_bar = 0;

% set D0
if init_ep
    D0 = 2*sqrt(2)*kappa*R0; 
else
    D0 = 2*sqrt(2)*kappa*R0 + (4*sqrt(2)*kappa*(2+sqrt(2*log(12/delta)))*(1+2*kappa)*R)/sqrt( min(p_hat,1-p_hat)*n - sqrt(2*n*log(12/delta)) );
end
mm = (n_p + n_n);
for i = 1:n
    p_hat = ((i+m-1)*p_hat+(y(i)==1))/(i+mm);
    
    % compute gradient
    % pred = v'*[X(:,i);0;0]; 
    pred = w'*X(:,i);
    temp = 2*((y(i)==-1)*p_hat-(y(i)==1)*(1-p_hat));
    ga = 2*(y(i)==1)*(1-p_hat)*(a-pred);
    gb = 2*(y(i)==-1)*p_hat*(b-pred);
    gw = ((1+alpha)*temp-ga-gb)*X(:,i);
    galpha = temp*pred-2*p_hat*(1-p_hat)*alpha;
    
    % update 
    % v = v-gamma*[gw;ga;gb]; %gv = [gw;ga;gb];
    w = w-gamma*gw;
    a = a-gamma*ga;
    b = b-gamma*gb;
    alpha = alpha+gamma*galpha;
    
    % projection
    for j = 1:K
        % v([1:d]) = ProjectOntoL1Ball(v([1:d]), R); % pro to omega1: w
        w = ProjectOntoL1Ball(w, R); % pro to omega1: w
        % v(d+1) = sign(v(d+1))*min(abs(v(d+1)),R*kappa); % pro to omega1: a
        a = sign(a)*min(abs(a),R*kappa); % pro to omega1: a
        % v(d+2) = sign(v(d+2))*min(abs(v(d+2)),R*kappa); % pro to omega1: b 
        b = sign(b)*min(abs(b),R*kappa); % pro to omega1: b 
        
        %%%%% v = v1+ProjectOntoL1Ball(v-v1,R0); % pro to l1 ball, B(v1,R0)
        % v_norm = norm(v-v1, 2);
        v_norm = sqrt(norm(w-w1,2)^2+(a-a1)^2+(b-b1)^2);
        if v_norm > R0
            % v = v1+v*R0/v_norm;
            tmp = R0/v_norm; 
            w = w1+tmp*w;
            a = a1+tmp*a;
            b = b1+tmp*b;
        end
    end
    
    % pro: alpha
    a_tmp = min([D0+alpha1,2*R*kappa,alpha]);
    alpha = max([-D0+alpha1,-2*R*kappa,a_tmp]);
    
    % average
    gamma_bar_old = gamma_bar;
    gamma_bar = gamma_bar + gamma;
    % v_bar = (gamma_bar_old*v_bar+gamma*v)/gamma_bar;
    w_bar = (gamma_bar_old*w_bar+gamma*w)/gamma_bar;
    a_bar = (gamma_bar_old*a_bar+gamma*a)/gamma_bar;
    b_bar = (gamma_bar_old*b_bar+gamma*b)/gamma_bar;
    alpha_bar = (gamma_bar_old*alpha_bar+gamma*alpha)/gamma_bar;
end

A_p = A_p + (X*(y==1))'; 
A_n = A_n + (X*(y==-1))';
n_p = n_p + sum(y==1); 
n_n = n_n + sum(y==-1); 
A = [A_n/n_n - A_p/n_p,0,0];
v_bar = [w_bar;a_bar;b_bar]; 
end
