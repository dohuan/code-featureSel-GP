function [forward_rank, backward_rank] = feature_ranking_1(...
                                             GPField,...                          
                                             data,...
                                             option)
%% Feature ranking for Crossvalidation modification on 09/17/14 by Huan Do
% NOTE: data: array of struct
nf = size(data(1).features,2);
n_fold = size(GPField,2);
forward_rank = zeros(1,nf);
backward_rank = zeros(n_fold,nf);

GPField_temp = GPField;
data_temp = data;
for index2 = 1 : nf
    % --- cross-validation ---
    for i=1:n_fold
        % --- drop the index2-th feature ---
        GPField_temp(i).Predicted_Layer_Map(:,:,index2)         = [];
        GPField_temp(i).Prediction_Layer_Variance(:,:,index2)   = [];
        GPField_temp(i).hyper_p(:,index2)                       = [];
        data_temp(i).features(:,index2)                         = [];
    end
    % --- cross-validation ---
    for i=1:n_fold
        test_fold = i;
        train_fold = 1:n_fold;
        train_fold(test_fold) = [];
        [~,rmse_temp(i,1),~] = localizing_1(GPField_temp(train_fold),...
                                            data_temp(test_fold).features,...
                                            data_temp(test_fold).location,...
                                            option);
        fprintf('Feature ranking for %d features, fold %d.\n',...
            nf,i);
    end
    backward_rank(:,index2) = rmse_temp;
    % --- reset the whole field and data with original features ---
    GPField_temp = GPField;
    data_temp = data;
end
backward_rank = mean(backward_rank,1);                                  
end