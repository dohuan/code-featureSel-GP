function [p0, accuracy,logL] = localizing_1( GPField,...                          
                                             features,...
                                             location,...
                                             option)   
%% Localization for Crossvalidation modification on 09/16/14 by Huan Do

%features = data.features;
%location = data.location;
n = size(features,1);
nf = size(features,2);
n_fold = size(GPField,2);

tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng1 = size(tmp1,2);
features_likelihood = [];
for i=1:n_fold
    ng = numel(GPField(i).Predicted_Layer_Map)/nf;
    map = reshape(GPField(i).Predicted_Layer_Map,ng,nf);
    map_var = reshape(GPField(i).Prediction_Layer_Variance,ng,nf);

    features_diffrence = (kron(ones(n,1),map) - kron(features,ones(ng,1))).^2;
    features_variance  = ones(n*ng,nf) * diag(GPField(i).hyper_p(4,:)) + ...
                           kron(ones(n,1),map_var);
    if (i==1)
        features_likelihood = -0.5*(log(features_variance) + ...
                            features_diffrence./features_variance + ...
                            ones(ng*n,nf)*2*pi);
    else
        features_likelihood = -0.5*(log(features_variance) + ...
                            features_diffrence./features_variance + ...
                            ones(ng*n,nf)*2*pi) ...
                          + features_likelihood;
    end
end
cost = sum(features_likelihood,2);              
total_cost = reshape(cost,ng,n);
[logL,I0] = max(total_cost);
logL = sum(logL);

I0 = I0-1;
IX = floor(I0/ng1)+1;    
IY = mod(I0,ng1)+1;
p0 = [tmp2(IX);tmp1(IY)]';
accuracy = sqrt(sum(sum((p0 - location(:,1:2)).^2))/n);
end