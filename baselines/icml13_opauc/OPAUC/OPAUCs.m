function w = OPAUCs(data,label,eta,lambda,tau)
%   The implementation of One-Pass AUC Optimization (high-dimensional dataset) in [2]. 
%   ========================================================================
%   Input:
%   OPAUC takes 4 input parameters
%
%   instance    a sparse matrix with size instanceNum * dimension. Each row vector 
%                 is a instance vector (not including their labels)
%
%   label       a vectore with size instanceNum, which correspends
%                 the instance's label
%
%   eta        the stepsize parameter
%
%   lamda      the regularization paremeter
%
%   tau        the parameter of nnz (number of non-zeros) 
%  ========================================================================
%   Output:
%   w          the learned linear classifier
%   ========================================================================
%  Reference:
%  [2] Wei Gao, Lu Wang, Rong Jin, Shenghou Zhu and Zhi-Hua Zhou. One-Pass AUC
%  Optimzation. Artificial Intelligence. In press.
[num,d]=size(data);
NumP=0;
NumN=0;
CenterP=zeros(1,d); % center of positive instances
CenterN=zeros(1,d); % center of negative instances
BP=sparse(d,d);   %approximation matrix of positive instances
BN=sparse(d,d);   %approximation matrix of negative instances
zeros_min=ceil(tau*d*0.9);
zeros_max=tau*d;   
w=sparse(1,d);
for t=1:num
    if label(t)==1
        NumP=NumP+1;
        CenterP=CenterP+(data(t,:)-CenterP)/NumP;
        BP=BP+data(t,:)'*data(t,:);
        if (nnz(BP)>zeros_max)
            max_=max(max(BP));min_=min(min(BP));
            mid=(max_+min_)/2;
            num_nonzeros=nnz(BP>mid);
            while or(num_nonzeros>zeros_max,num_nonzeros<zeros_min)
                if num_nonzeros>zeros_max
                    min_=mid;
                    mid=(max_+min_)/2;
                else
                    max_=mid;
                    mid=(max_+min_)/2;
                end
                if max_-min_<1e-5
                    mid=max_;
                    break
                end
                num_nonzeros=nnz(BP>mid);
            end
            BP=BP.*(BP>mid);
        end
        if NumN>0
            w=w-eta*(lambda*w-data(t,:)+CenterN+w*(data(t,:)-CenterN)'*(data(t,:)-CenterN) ...,
                +w*BN/NumN-(w*CenterN')*CenterN);
        end
    else
        NumN=NumN+1;
        CenterN=CenterN+(data(t,:)-CenterN)/NumN;
        BN=BN+data(t,:)'*data(t,:);
        if (nnz(BN)>zeros_max)
            max_=max(max(BN));min_=min(min(BN));
            mid=(max_+min_)/2;
            num_nonzeros=nnz(BN>mid);
            while or(num_nonzeros>zeros_max,num_nonzeros<zeros_min)
                if num_nonzeros>zeros_max
                    min_=mid;
                    mid=(max_+min_)/2;
                else
                    max_=mid;
                    mid=(max_+min_)/2;
                end
                if max_-min_<1e-5
                    mid=max_;
                    break
                end
                num_nonzeros=nnz(BN>mid);
            end
            BN=BN.*(BN>mid);
        end
        if NumP>0
            w=w-eta*(lambda*w+data(t,:)-CenterP+(w*(data(t,:)-CenterP)')*(data(t,:)-CenterP) ...,
                +w*BP/NumP-(w*CenterP')*CenterP);
        end
    end
end