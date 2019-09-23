function kappa = get_kappa(X)
% kappa = sup(X) by l2 norm where X is d*n 
% January, 2018
% xzhang131@uiowa.edu
% version 1.0
    [d,n] = size(X);
    kappa = max(sqrt(ones(1,d)*(X.*X))); 
end