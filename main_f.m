function output = main_f(features)
%%                        Modified "main_function.m"
%                       Author: Huan N. Do Sep-26-14
%%
global system
system.sigma2w = 0.5;
if isfield(features,'folderpath')
    folderpath = features.folderpath;
    if strcmp(features.data_type,'in')
        mode_temp = 0;
    elseif strcmp(features.data_type,'out_unfix')
        mode_temp = 1;
    elseif strcmp(features.data_type,'out_fix')
        mode_temp = 1;
    end
else
    if strcmp(features.data_type,'in')
        mode_temp = 0;
        folderpath = './images2/';
    elseif strcmp(features.data_type,'out_unfix')
        mode_temp = 1;
        folderpath = './imagesAcuraRun/merged(K_L)/';
    elseif strcmp(features.data_type,'out_fix')
        mode_temp = 1;
        folderpath = './imagesAcuraRun/merged(K_L_fixedAngle)/';
    end
end
output.name = [features.name '-' ...
               num2str(features.nf) '-' ...
               features.data_type];
[train_location, train_features] = ...
    extract_features_large(features.name,...
    features.nf,...
    mode_temp,...
    folderpath);

if (mode_temp==0)
    option.x_limit = [1 23];     % The limit on the x cordinate will be set here
    option.y_limit = [1 9];
    option.resolution = 0.2;    % The resulotion of predicted map will be determined here
else
    min_loc = min(min(train_location(:,1)),min(train_location(:,2)));
    max_loc = max(max(train_location(:,1)),max(train_location(:,2)));
    option.x_limit = [min(train_location(:,1)) max(train_location(:,1))];     % The limit on the x cordinate will be set here
    option.y_limit = [min(train_location(:,2)) max(train_location(:,2))];
    option.resolution = round((max_loc-min_loc)/100);
end

nt = size(train_location,1);
crossvalidation_index = 1:5:nt; % we choose 10 percent with jumping 10 step each time
test_index = crossvalidation_index + 1; % we keep 10% cross validation data
test_features = train_features(test_index,:);
test_location = train_location(test_index,:);
crossvalidation_features = train_features(crossvalidation_index,:);
crossvalidation_location = train_location(crossvalidation_index,:);

single_sample_point = train_location(1,1:2);

train_features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset

tmp1 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp2 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng1 = size(tmp1,2);
ng2 = size(tmp2,2);
[S1,S2] = meshgrid(tmp1,tmp2);
S = [S1(:), S2(:)];

% --- Config GPML ---
covfunc = @covSEard;
likfunc = @likGauss;

nf = size(train_features,2);
hyper_p = zeros(4,nf);
X = train_location(:,1:2);

tic
for index = 1:nf
    % --- Learning hyper-parameters by ML ---
    if mode_temp==0
        hyp(index).cov = log([1; 1; var(train_features(:,index))]);
        hyp(index).lik = log(0.5);
    else
        hyp(index).cov = log([1; 1; var(train_features(:,index))]);
        hyp(index).lik = log(0.1);
    end
%     p0 = [var(train_features(:,index)) 5 5 0.01];
%     [hyper_p(:,index), ~] = ...
%         HyperParameter(train_features(:,index),train_location(:,1:2)); %extract hyper parameters for each layer sepratly
%     str = sprintf('hyper-parameters # %d: \t (sig_f^2: %0.2f, \t sig_x: %0.2f, \t sig_y: %0.2f, \t sig_w^2: %0.2f)',index,hyper_p(1,index),hyper_p(2,index),hyper_p(3,index),hyper_p(4,index));
%     disp(str)
    
    y = train_features(:,index);
    hyp(index) = minimize(hyp(index), @gp, -500, @infExact, [], covfunc, likfunc, X, y);
    %hyp(index).cov = log(hyper_p(1:3,index));
    %hyp(index).lik = log(hyper_p(4,index));
    % --- Learing GP fields ---
    nlml = gp(hyp(index), @infExact, [], covfunc, likfunc, X, y);
    [est , Variance] = ...
        gp(hyp(index), @infExact, [], covfunc, likfunc, X, y,S);
    Predicted_map(:,:,index) = reshape(est,ng2,ng1);
    Predicted_variance(:,:,index) = reshape(Variance,ng2,ng1);
    hyper_p(1:3,index) = exp(hyp(index).cov);
    hyper_p(4,index) = exp(hyp(index).lik);
    progressbar(index/nf)
end
collapsedTime = toc;
output.learning_hyper_time = collapsedTime;
option.hyperparametrs = hyper_p;
output.test = test_location;
save('Layer-Map.mat',...
    'Predicted_map',...
    'Predicted_variance',...
    'option',...
    'test_features',...
    'crossvalidation_features');

%% Calculate RMSE when using ALL features
selected_nf = 1;
selected_featuters = (1:nf);
progressbar('Feature elemination ...')

% --- all features applied on TEST set
[output.all_t, output.rmse_all_t,logL_all] = localizing( ...
    Predicted_map,...                                % required Mapping mean
    Predicted_variance,...                          % required Variance
    test_features,...                                           % measured feature
    test_location,...
    option);

% --- all features applied on VALIDATION set
[output.all_v, output.rmse_all_v,~] = localizing( ...
    Predicted_map,...                                % required Mapping mean
    Predicted_variance,...                          % required Variance
    crossvalidation_features,...                                           % measured feature
    crossvalidation_location,...
    option);
%%                       --- Backward elimination ---
tic
for index1 = 1:(nf-selected_nf)
    [~, backward_rank] = ...
        feature_ranking( Predicted_map,...
        Predicted_variance, ...
        crossvalidation_features, ...
        crossvalidation_location, option);
    
    [CX(index1), IX] = min(backward_rank);
    
    Predicted_map(:,:,IX)                = [];
    Predicted_variance(:,:,IX)           = [];
    crossvalidation_features(:,IX)       = [];
    test_features(:,IX)                  = [];
    option.hyperparametrs(:,IX)          = [];
    [~,IY] = sort(selected_featuters);
    selected_featuters(IY(IX))= 1000 - index1;
    progressbar(index1/(nf-selected_nf))
end
[~,index1]=min(CX);
output.opt_features = nf - index1;
[~,B] = sort(selected_featuters);
IX = B(1:output.opt_features);

collapsedTime = toc;
output.BE_time = collapsedTime;


load Layer-Map % REFRESH map and option
save(['./main_f/' features.name '-' features.data_type '-data.mat'],...
    'IX',...
    'Predicted_map',...
    'Predicted_variance',...
    'option');

option.hyperparametrs = option.hyperparametrs(:,IX);
[output.BE_test, output.rmse_selected_test,logL_opt] = localizing( ...
    Predicted_map(:,:,IX),...                                % required Mapping mean
    Predicted_variance(:,:,IX),...                          % required Variance
    test_features(:,IX),...                                           % measured feature
    test_location,...
    option);

tic
load Layer-Map % REFRESH map and option
option.hyperparametrs = option.hyperparametrs(:,IX);
[output.BE_validation, output.rmse_selected_valid] = localizing( ...
    Predicted_map(:,:,IX),...                                % required Mapping mean
    Predicted_variance(:,:,IX),...                          % required Variance
    crossvalidation_features(:,IX),...                                           % measured feature
    crossvalidation_location,...
    option);
collapsedTime = toc;
output.allset_test_estimation_time = collapsedTime;

% --- Localization time for ONE test point, includes features extraction
test_img = imread([folderpath '1-d.jpg']);
[~,localization_time] = localize_me_app(features,...
    test_img,...
    single_sample_point);
output.single_test_estimation_time = localization_time;

% --- Calculate the BIC index for All and Selected cases ---
output.BIC_all = -2*(logL_all) + 3*size(train_features,2)*log(size(train_features,1));
output.BIC_selected = -2*(logL_opt) + 3*output.opt_features*log(size(train_features,1));

end
