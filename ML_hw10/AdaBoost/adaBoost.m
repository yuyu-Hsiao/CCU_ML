%%
% File Name: AdaBoost
% This is the implementation of the ada boost algorithm.
% Parameters - very easy to gues by name...
% Return values: i - hypothesis-index  vector.
%                t - threshhols vector
%                beta - weighted beta.
%%
function boosted=adaBoost(train,train_label,cycles)
    d=size(train);
	distribution=ones(1,d(1))/d(1); %u
	error=zeros(1,cycles);
	beta=zeros(1,cycles); 
	label=(train_label(:)>=5);
	for j=1:cycles
	    [i,t]=weakLearner(distribution,train,label);
        error(j)=distribution*abs(label-(train(:,i)>=t));
        beta(j)=error(j)/(1-error(j));
        boosted(j,:)=[beta(j),i,t];
        
        distribution=distribution.* exp(log(beta(j))*(1-abs(label-(train(:,i)>=t ))))'; 
        distribution=distribution/sum(distribution); %normalize to 0~1
  
end
    
    