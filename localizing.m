function [p0, accuracy,logL] = localizing( ...
                     Predicted_Layer_Map,...                                % required Mapping mean
                     Prediction_Layer_Variance,...                          % required Variance
                     features,...                                           % measured feature
                     location,...
                     option)                                                % The noise of Observations
% This Function is written in 4/25/2013 by Mahdi Jadaliha in order to use 
% Maximum a posteriori estimation (MAP) to estimate location of observations 
% according the provided map and variance information.
%   Detailed explanation goes here

n = size(features,1);
nf = size(features,2);
ng = numel(Predicted_Layer_Map)/nf;
map = reshape(Predicted_Layer_Map,ng,nf);
map_var = reshape(Prediction_Layer_Variance,ng,nf);

tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng1 = size(tmp1,2);

features_diffrence = (kron(ones(n,1),map) - kron(features,ones(ng,1))).^2;
features_variance  = ones(n*ng,nf) * diag(option.hyperparametrs(4,:)) + ...
                       kron(ones(n,1),map_var);
features_likelihood= log(features_variance) + ...
                        features_diffrence./features_variance;

features_likelihood_2 = -0.5*(log(features_variance) + ...
                        features_diffrence./features_variance + ...
                        ones(ng*n,nf)*log(2*pi));                    
cost = sum(features_likelihood,2);
cost_2 = sum(features_likelihood_2,2);
% Compute RMS Error using all predicted map
total_cost = reshape(cost,ng,n);
total_cost_2 = reshape(cost_2,ng,n);
%total_cost_sum = ones(ng,1)*log(sum(exp(total_cost_2),1));
%total_cost_sum = ones(ng,1)*sum(total_cost_2,1);
%total_cost_2 = total_cost_2 - total_cost_sum;
[~,I0] = min(total_cost);
[logL,~] = max(total_cost_2);
logL = sum(logL);

I0 = I0-1;
IX = floor(I0/ng1)+1;    
IY = mod(I0,ng1)+1;
p0 = [tmp2(IX);tmp1(IY)]';
accuracy = sqrt(sum(sum((p0 - location(:,1:2)).^2))/n);
%logL = sum(logL);
end
