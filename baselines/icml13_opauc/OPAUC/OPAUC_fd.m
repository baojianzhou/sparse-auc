function w = OPAUC_fd(data,label,eta,lambda,tau)
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
%   tau        the sketch-size parameter
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
CenterP=zeros(1,d);
CenterN=zeros(1,d);
BP=zeros(tau,d);
BN=zeros(tau,d);
t_num=ceil(tau*0.9);
BP_zeros_index=1;
BN_zeros_index=1;
w=sparse(1,d);
for t=1:num
    if label(t)==1
        NumP=NumP+1;
        CenterP=CenterP+(data(t,:)-CenterP)/NumP;
        BP(BP_zeros_index,:)=data(t,:);
        BP_zeros_index=BP_zeros_index+1;
        % update BP
        if BP_zeros_index>tau
            [U,S,V]=svd(BP,'econ');
            for i=1:t_num-1
                BP(i,:)=sqrt(S(i,i)^2-S(t_num,t_num)^2)*V(:,i)';
            end
            BP(t_num:tau,:)=0;
            index=t_num-1;
            while sum(BP(index,:))==0 
                index=index-1;
            end
            BP_zeros_index=index+1;
        end
        if NumN>0
            w=w-eta*(lambda*w-data(t,:)+CenterN+w*(data(t,:)-CenterN)'*(data(t,:)-CenterN) ...,
                +(w*BN')*BN/NumN-w*CenterN'*CenterN);
        end
    else
        NumN=NumN+1;
        CenterN=CenterN+(data(t,:)-CenterN)/NumN;
        BN(BN_zeros_index,:)=data(t,:);
        BN_zeros_index=BN_zeros_index+1;
        % update BN
        if BN_zeros_index>tau
            [U,S,V]=svd(BN,'econ');
            for i=1:t_num-1
                BN(i,:)=sqrt(S(i,i)^2-S(t_num,t_num)^2)*V(:,i)';
            end
            BN(t_num:tau,:)=0;
            index=t_num-1;
            while sum(BN(index,:))==0 
                index=index-1;
            end
            BN_zeros_index=index+1;
        end
        if NumP>0
            w=w-eta*(lambda*w+data(t,:)-CenterP+(w*(data(t,:)-CenterP)')*(data(t,:)-CenterP) ...,
                +(w*BP')*BP/NumP-(w*CenterP')*CenterP);
        end
    end
end