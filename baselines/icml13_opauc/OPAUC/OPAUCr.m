function w=OPAUCr(instance,label,eta,lambda,tau)
%  The implementation of One-Pass AUC Optimization (high-dimensional dataset) in [1]. 
%  ========================================================================
%  Input:
%  OPAUC takes 4 input parameters
%
%    instance   a matrix with size instanceNum * dimension. Each row vector 
%                 is a instance vector (not including their labels)
%
%    label      a vectore with size instanceNum, which correspends
%                 the instance's label
%
%    eta        the stepsize parameter
%
%    tau        the rank parameter
%
%    lamda      the regularization paremeter
%  ========================================================================
%  Output:
%    w          the learned linear classifier
%  ========================================================================
%  Reference:
%  [1] Wei Gao, Rong Jin, Shenghou Zhu and Zhi-Hua Zhou. One-Pass AUC
%  Optimzation. In: Proceedings of the 30th International Conference on
%  Machine Learning (ICML'13), Atlanta, GA, 2013, JMLR: W&CP 28(3),
%  pp.906-914.
[num,d]=size(instance);
NumP=0;
NumN=0;
CenterP=sparse(1,d);
CenterN=sparse(1,d);
hCenterP=sparse(1,tau);
hCenterN=sparse(1,tau);
ZP=sparse(d,tau);
ZN=sparse(d,tau);
w=sparse(1,d);
for t=1:num
    at=randn(1,tau)/sqrt(tau);
    if label(t)==1
        NumP=NumP+1;
        CenterP=CenterP+(instance(t,:)-CenterP)/NumP;
        hCenterP=hCenterP+(at-hCenterP)/NumP;
        ZP=ZP+instance(t,:)'*at;
        if NumN>0
            w=w-eta*(lambda*w-instance(t,:)+CenterN+w*(instance(t,:)-CenterN)'*(instance(t,:)-CenterN) ...,
                +w*ZN*ZN'/NumN-w*CenterN'*hCenterN*hCenterN'*CenterN);
        end
    else
        NumN=NumN+1;
        CenterN=CenterN+(instance(t,:)-CenterN)/NumN;
        hCenterN=hCenterN+(at-hCenterN)/NumN;
        ZN=ZN+instance(t,:)' * at;
        if NumP>0
            w=w-eta*(lambda*w+instance(t,:)-CenterP+(w*(instance(t,:)-CenterP)')*(instance(t,:)-CenterP) ...,
                +(w*ZP)*ZP'/NumP-(w*CenterP')*hCenterP*hCenterP'*CenterP);
        end
    end
end
return;