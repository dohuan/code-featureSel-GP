
function output = main_function(features)
global system
system.sigma2w = 0.5;
%% ------------------ Load data and doing pre processings -----------------
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
        folderpath = './data/images2/';
    elseif strcmp(features.data_type,'out_unfix')
        mode_temp = 1;
        folderpath = './data/imagesAcuraRun/merged(K_L)/';
    elseif strcmp(features.data_type,'out_fix')
        mode_temp = 1;
        folderpath = './data/imagesAcuraRun/merged(K_L_fixedAngle)/';
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

% --- adding noise to location here, ONLY for Case 1 nosy sampling loc ---
%noiseStd = 1;
%noiseAddedX = noiseStd*randn(size(train_location,1),1);
%noiseAddedY = noiseStd*randn(size(train_location,1),1);
%train_location(:,1) = train_location(:,1) + noiseAddedX;
%train_location(:,2) = train_location(:,2) + noiseAddedY;

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

% -- seperate test set
crossvalidation_index = 1:5:nt; % we choose 10 percent with jumping 10 step each time
test_index = crossvalidation_index + 1; % we keep 10% cross validation data
test_features = train_features(test_index,:);
test_location = train_location(test_index,:);
crossvalidation_features = train_features(crossvalidation_index,:);
crossvalidation_location = train_location(crossvalidation_index,:);

single_sample_point = train_location(1,1:2);

train_features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
tic
% -- hyper-parameter estimated by ML --
nf = size(train_features,2);
hyper_p = zeros(4,nf);
logLikelihood = zeros(nf,1);

% --- Config GPML ---
%covfunc = @covSEard;
%likfunc = @likGauss;

for index = 1:nf
    p0 = [var(train_features(:,index)) 5 5 0.01];
    [hyper_p(:,index), logLikelihood(index)] = ...
        HyperParameter(train_features(:,index),train_location(:,1:2)); %extract hyper parameters for each layer sepratly
%     hyp(index).cov = log([1; 1; var(train_features(:,index))]);
%     hyp(index).lik = log(0.5);
%     hyp(index) = minimize(hyp(index), @gp, -500, @infExact, [], ...
%          covfunc, likfunc, train_location(:,1:2), train_features(:,index));
%     hyper_p(1:3,index) = exp(hyp(index).cov);
%     hyper_p(4,index) = exp(hyp(index).lik);
    
    str = sprintf('hyper-parameters # %d: \t (sig_f^2: %0.2f, \t sig_x: %0.2f, \t sig_y: %0.2f, \t sig_w^2: %0.2f)',index,hyper_p(1,index),hyper_p(2,index),hyper_p(3,index),hyper_p(4,index));
    disp(str)
    progressbar(index/nf)
end

% train_features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
% train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset


%% -- plot training data set
% handle_fig1 = figure('Name','locations:','NumberTitle','off');
% axes1 = plot3(train_location(:,1),train_location(:,2),train_location(:,3),'MarkerFaceColor',[1 1 0],'MarkerEdgeColor',[1 0 1],...
%     'Marker','o',...
%     'LineWidth',2); % plot training sampling positions along with extracted features
% % set(axes1,'ZTickLabel',{},'ZTick',zeros(1,0));
% zlim([-1000 1000]);
% zlim('manual')
% xlabel('x direction (m)')
% ylabel('y direction (m)')
% hdt = datacursormode;
% set(hdt,'DisplayStyle','window');
% set(hdt,'UpdateFcn',{@labeldtips,train_features});

%% Compute mean and variance fields

tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng2 = size(tmp2,2);
ng1 = size(tmp1,2);
Predicted_Layer_Map = zeros(ng1,ng2,nf);
Prediction_Layer_Variance = zeros(ng1,ng2,nf);
progressbar('Computing Map layers ...')
% Produce grid points
[x,y] = meshgrid(tmp2,tmp1);
xstar = [reshape(x,[],1),reshape(y,[],1)];
X = train_location(:,1:2);
for index =1:nf;
    Y = train_features(:,index);
    sigma2w = hyper_p(4,index);
    n = size(X,1);
    system.Sigma2X = diag([hyper_p(2,index)^2,hyper_p(3,index)^2]);
    system.sigma2f = hyper_p(1,index);
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
    
    progressbar(index/nf)
end

collapsedTime = toc;
output.hyper_field_estimation_time = collapsedTime;
%plotpredictedmaps(Predicted_Layer_Map,option);
%plotpredictedmaps(Prediction_Layer_Variance,option);

option.hyperparametrs = hyper_p;
output.test = test_location;
save('Layer-Map.mat','Predicted_Layer_Map','Prediction_Layer_Variance','option','test_features','crossvalidation_features');
%% Calculate RMSE when using ALL features
selected_nf = 1;
selected_featuters = (1:nf);
progressbar('Feature elemination ...')

% --- all features applied on TEST set
[output.all, output.rmse_all,logL_all] = localizing( ...
    Predicted_Layer_Map,...                                % required Mapping mean
    Prediction_Layer_Variance,...                          % required Variance
    test_features,...                                           % measured feature
    test_location,...
    option);

% --- all features applied on VALIDATION set
[output.all_v, output.rmse_all_v,~] = localizing( ...
    Predicted_Layer_Map,...                                % required Mapping mean
    Prediction_Layer_Variance,...                          % required Variance
    crossvalidation_features,...                                           % measured feature
    crossvalidation_location,...
    option);

% IX = (1:16); % supposed to be 1:16, Mahdi tried different values
% option_tmp = option;
% option_tmp.hyperparametrs = option.hyperparametrs(:,IX);
% 
% [~, output.results(2),~] = localizing( ...
%     Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
%     Prediction_Layer_Variance(:,:,IX),...                          % required Variance
%     test_features(:,IX),...                                           % measured feature
%     test_location,...
%     option_tmp);
%%                       --- Backward elimination ---
tic
for index1 = 1:(nf-selected_nf)
    [~, backward_rank] = ...
        feature_ranking( Predicted_Layer_Map,...
        Prediction_Layer_Variance, ...
        crossvalidation_features, ...
        crossvalidation_location, option);
    
    [CX(index1), IX] = min(backward_rank);
    
    Predicted_Layer_Map(:,:,IX)          = [];
    Prediction_Layer_Variance(:,:,IX)    = [];
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
save(['./result/' features.name '-' features.data_type '-data.mat'],...
    'IX',...
    'Predicted_Layer_Map',...
    'Prediction_Layer_Variance',...
    'option');

option.hyperparametrs = option.hyperparametrs(:,IX);
[output.BE_test, output.rmse_selected_test,logL_opt] = localizing( ...
    Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
    Prediction_Layer_Variance(:,:,IX),...                          % required Variance
    test_features(:,IX),...                                           % measured feature
    test_location,...
    option);

tic
load Layer-Map % REFRESH map and option
option.hyperparametrs = option.hyperparametrs(:,IX);
[output.BE_validation, output.rmse_selected_valid] = localizing( ...
    Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
    Prediction_Layer_Variance(:,:,IX),...                          % required Variance
    crossvalidation_features(:,IX),...                                           % measured feature
    crossvalidation_location,...
    option);
collapsedTime = toc;
output.test_estimation_time = collapsedTime;

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
