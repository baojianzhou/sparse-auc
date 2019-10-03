function w=OPAUC(instance,label,eta,lambda)
%  The implementation of One-Pass AUC Optimization in [1]. 
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
%    lamda      the regularization paremeter
%  ========================================================================
%  Output:
%    w          the learned linear classifier
%  ========================================================================
%  Reference:
%  [1] Wei Gao, Rong Jin, Shenghou Zhu and Zhi-Hua Zhou. One-Pass AUC
%  Optimzation. In: Proceedings of the 30th International Conference on
%  Machine Learning (ICML'13), Atlanta, GA, 2013, JMLR: W&CP 28(3), pp.906-914.
[num,d]=size(instance);
NumP=0;
NumN=0;
CenterP=zeros(1,d);
CenterN=zeros(1,d);
CovMatP=zeros(d,d);
CovMatN=zeros(d,d);
w=zeros(1,d);
for t=1:num
    if label(t)==1
        NumP=NumP+1;
        tmp=CenterP;
        CenterP=CenterP+(instance(t,:)-CenterP)/NumP;
        CovMatP=CovMatP+(instance(t,:)' * instance(t,:)-CovMatP-tmp'*tmp)/NumP+tmp'*tmp-CenterP'*CenterP;
        if NumN>0
            w=w-eta*(lambda*w-instance(t,:)+CenterN+w*(instance(t,:)-CenterN)'*(instance(t,:)-CenterN)+w*CovMatN);
        end
    else
        NumN=NumN+1;
        tmp=CenterN;
        CenterN=CenterN+(instance(t,:)-CenterN)/NumN;
        CovMatN=CovMatN+(instance(t,:)' * instance(t,:)-CovMatN-tmp'*tmp)/NumN+tmp'*tmp-CenterN'*CenterN;
        if NumP>0
            w=w-eta*(lambda*w+instance(t,:)-CenterP+w*(instance(t,:)-CenterP)'*(instance(t,:)-CenterP)+w*CovMatP);
        end
    end
end
return;