function output = main_function_1(folderpath,featureMode,k_fold)
%%                       Run with the file tableMaker.m
%                           Using Cross-Validation
% ---------------------- variables declaration ----------------------------
tic
global system
system.sigma2w = 0.5;
%% ------------------ Load data and doing pre processings -----------------
%%  for outdoor dataset with a SINGLE dataset
[train_location, train_features] = ...
    extract_features(featureMode.name,...
    featureMode.featureNumber,...
    folderpath.mode,...
    folderpath.path);

if (folderpath.mode == 0)
    option.x_limit = [1 23];     % The limit on the x cordinate will be set here
    option.y_limit = [1 9];
    option.resolution = 0.2;    % The resulotion of predicted map will be determined here
else
    option.x_limit = [min(train_location(:,1)) max(train_location(:,1))];     % The limit on the x cordinate will be set here
    option.y_limit = [min(train_location(:,2)) max(train_location(:,2))];
    option.resolution = ...
        round((max(train_location(:,1)) - min(train_location(:,1)))/100);
end

nt = size(train_location,1);

test_index = 1:5:nt;
test_features = train_features(test_index,:);
test_location = train_location(test_index,:);
train_features(test_index,:) = [];
train_location(test_index,:) = [];


nt = size(train_location,1); % update post-splited train data size
nf = size(train_features,2); % number of total features
% --------------------- create k-fold data split --------------------------
count = 0;
fold_no = 1;
%k_fold = 5;
count_limit = round(size(train_location,1)/k_fold);
for i=1:size(train_location,1)
    if (count<count_limit)
        CV_index(i,1) = fold_no;
        count = count + 1;
    else
        if (fold_no<k_fold)
            fold_no = fold_no + 1;
        end
        CV_index(i,1) = fold_no;
        count = 1;
    end
end

% ------------ Estimate hyper-parameter for each k-fold -------------------
for i=1:k_fold
    fold_IX = find(CV_index==i);
    for index=1:nf
        p0 = [var(train_features(:,index)) 5 5 0.01];
        [hyper_p_temp(:,index),~] = ...
            HyperParameter(train_features(fold_IX,index),...
            train_location(fold_IX,1:2));
        fprintf('Calculating hyperparameters for fold %d: %d%% \n',...
            i,round(index/nf*100));
    end
    GP_field(i).hyper_p = hyper_p_temp;
    Data(i).location = train_location(fold_IX,1:2);
    Data(i).features = train_features(fold_IX,:);
end

% ------------- Construct GP mean and covariance fields -------------------
tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng2 = size(tmp2,2);
ng1 = size(tmp1,2);
for i=1:k_fold
    fold_IX = find(CV_index==i);
    Predicted_Layer_Map = zeros(ng1,ng2,nf);
    Prediction_Layer_Variance = zeros(ng1,ng2,nf);
    [x,y] = meshgrid(tmp2,tmp1);
    xstar = [reshape(x,[],1),reshape(y,[],1)];
    X = train_location(fold_IX,1:2);
    for index =1:nf;
        Y = train_features(fold_IX,index);
        sigma2w = GP_field(i).hyper_p(4,index);
        n = size(X,1);
        system.Sigma2X = diag([GP_field(i).hyper_p(2,index)^2,...
                               GP_field(i).hyper_p(3,index)^2]);
        system.sigma2f = GP_field(i).hyper_p(1,index);
        KxstarX = CovarianceMatrix(xstar,X);
        KXX = CovarianceMatrix(X);
        Kxstarxstar = CovarianceMatrix(xstar);
        Lambda = (KXX+sigma2w*eye(n))^(-1);
        zhatxstar = KxstarX*Lambda*Y;
        sigma2xstar = Kxstarxstar - KxstarX * Lambda * KxstarX';
        
        sigma2xstar = diag(sigma2xstar);
        zhatxstar = reshape(zhatxstar,ng1,ng2,[]);
        sigma2xstar = reshape(sigma2xstar,ng1,ng2,[]);
        
        Predicted_Layer_Map(:,:,index) = zhatxstar;
        Prediction_Layer_Variance(:,:,index) = sigma2xstar;
        
        fprintf('Calculating GP field for fold %d: %d%% \n',...
            i,round(index/nf*100));
    end
    GP_field(i).Predicted_Layer_Map = Predicted_Layer_Map;
    GP_field(i).Prediction_Layer_Variance = Prediction_Layer_Variance;
end
save('GP-data.mat','GP_field','Data');
% ------------- Backward Elimination + Cross-validation -------------------
selected_nf = 1;
selected_features = (1:nf);
for index1 = 1:(nf-selected_nf)
    [~,backward_rank] = feature_ranking_1(GP_field,Data,option);
    [CX(index1), IX] = min(backward_rank);
    % --- drop the feature with lowest backward rank ---
    
    for i=1:k_fold
        GP_field(i).Predicted_Layer_Map(:,:,IX) = [];
        GP_field(i).Prediction_Layer_Variance(:,:,IX) = [];
        GP_field(i).hyper_p(:,IX) = [];
        Data(i).features(:,IX) = [];
    end
    
    [~,IY] = sort(selected_features);
    selected_features(IY(IX))= 1000 - index1;
    fprintf('Backward elimination: %d%%\n',round(index1/(nf-selected_nf*100)));
end
[~,index_opt]=min(CX);
output.opt_feature = nf - index_opt;
[~,B] = sort(selected_features);
IX_survival = B(1:output.opt_feature);

% ----------------------- RMSE using ALL features -------------------------
load GP-data.mat
[output.path_All,output.rmse_all] = localizing_1(...
                                        GP_field,...
                                        test_features,...
                                        test_location,...
                                        option);
% ---------------------- RMSE using SELECTED features ---------------------
load GP-data.mat
for i=1:k_fold
    GP_field(i).Predicted_Layer_Map = ...
                          GP_field(i).Predicted_Layer_Map(:,:,IX_survival);
    GP_field(i).Prediction_Layer_Variance = ...
                    GP_field(i).Prediction_Layer_Variance(:,:,IX_survival);
    GP_field(i).hyper_p = GP_field(i).hyper_p(:,IX_survival);
    Data(i).features = Data(i).features(:,IX_survival);
end
test_features = test_features(:,IX_survival);

[output.path_Selected,output.rmse_selected] = localizing_1(...
                                        GP_field,...
                                        test_features,...
                                        test_location,...
                                        option);
timeCollapsed = toc;
fprintf('Collapsed time: %d minutes',timeCollapsed/60);
end

