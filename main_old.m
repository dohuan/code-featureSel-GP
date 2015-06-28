clear
clc
close all
tic
mode = 1; % 0 for indoor dataset, 1 for outdoor dataset single dataset, 2 for outdoor TRAIN-TEST dataset

global system
system.sigma2w = 0.5;       % The measurements noise will be set here
%% ------------------ Load data and doing pre processings -----------------
switch (mode)
    case 0
    %%  for indoor dataset
        folderpath = '.\images2\';
        option.x_limit = [1 23];     % The limit on the x cordinate will be set here
        option.y_limit = [1 9];
        option.resolution = 0.2;    % The resulotion of predicted map will be determined here
        [train_location, train_features] = ...
                            extract_features('ABS_FFT',64,mode,folderpath);     % Choices are {ABS_FFT, WAVELET, LOAD,HISTOGRAM}
        nt = size(train_location,1);

        % -- seperate test set
        test_index = 1:10:nt; % we choose 10 percent with jumping 10 step each time
        crossvalidation_index = test_index + 1; % we keep 10% cross validation data
        test_features = train_features(test_index,:);
        test_location = train_location(test_index,:);
        crossvalidation_features = train_features(crossvalidation_index,:);
        crossvalidation_location = train_location(crossvalidation_index,:);

        train_features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
        train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
        
        % -- hyper-parameter estimated by ML --
        logLikelihood = zeros(nf,1);
        for index = 1:nf
            p0 = [var(train_features(:,index)) 5 5 0.01];
            [hyper_p(:,index), logLikelihood(index)] = ...
                HyperParameter(train_features(:,index),train_location(:,1:2)); %extract hyper parameters for each layer sepratly
            str = sprintf('hyper-parameters # %d: \t (sig_f^2: %0.2f, \t sig_x: %0.2f, \t sig_y: %0.2f, \t sig_w^2: %0.2f)',index,hyper_p(1,index),hyper_p(2,index),hyper_p(3,index),hyper_p(4,index));
            disp(str)
            progressbar(index/nf)
        end

    case 1
    %%  for outdoor dataset with a SINGLE dataset
        %folderpath = './imagesAcuraRun/merged(O_P)/';
        %folderpath = './imagesAcuraRun/merged(O_P_fixedAngle)/';
        folderpath = './imagesAcuraRun/merged(K_L_fixedAngle)/';
        [train_location, train_features] = ...
                            extract_features('ABS_FFT',32,mode,folderpath);     % Choices are {ABS_FFT, WAVELET, LOAD,HISTOGRAM}
        option.x_limit = [min(train_location(:,1)) max(train_location(:,1))];     % The limit on the x cordinate will be set here
        option.y_limit = [min(train_location(:,2)) max(train_location(:,2))];
        option.resolution = round((max(train_location(:,1)) - min(train_location(:,1)))/100);
        nt = size(train_location,1);

        % -- seperate test set
        crossvalidation_index = 1:5:nt;
        test_index = crossvalidation_index + 1;
        test_features = train_features(test_index,:);
        test_location = train_location(test_index,:);
        crossvalidation_features = train_features(crossvalidation_index,:);
        crossvalidation_location = train_location(crossvalidation_index,:);
        
        train_features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
        train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
        
        % -- hyper-parameter estimated by ML --
        nf = size(train_features,2);
        hyper_p = zeros(4,nf);
        logLikelihood = zeros(nf,1);
        for index = 1:nf
            p0 = [var(train_features(:,index)) 5 5 0.01];
            [hyper_p(:,index), logLikelihood(index)] = ...
                HyperParameter(train_features(:,index),train_location(:,1:2)); %extract hyper parameters for each layer sepratly
            str = sprintf('hyper-parameters # %d: \t (sig_f^2: %0.2f, \t sig_x: %0.2f, \t sig_y: %0.2f, \t sig_w^2: %0.2f)',index,hyper_p(1,index),hyper_p(2,index),hyper_p(3,index),hyper_p(4,index));
            disp(str)
            progressbar(index/nf)
        end
    case 2
    %%  for outdoor dataset with separated TRAIN-TEST datasets
        %folderpath = '.\imagesBoatRun\';                   % The folderpath of refrence file will be here
        testStartIndex = 171;
        folderpath = './imagesAcuraRun/merged(O_P)/';
        [train_location, train_features, hyper_p] = ...
            extract_features('WAVELET',72,mode,folderpath);     % Choices are {ABS_FFT, WAVELET, LOAD,HISTOGRAM}
        

        option.x_limit = [min(train_location(:,1)) max(train_location(:,1))];     % The limit on the x cordinate will be set here
        option.y_limit = [min(train_location(:,2)) max(train_location(:,2))];
        option.resolution = round((max(train_location(:,1)) - min(train_location(:,1)))/100);
        
        test_location = train_location(testStartIndex:end,:);
        test_features = train_features(testStartIndex:end,:);
        
%         downSampleIndex = 1:5:size(test_location,1);
%         test_location = test_location(downSampleIndex,:);
%         test_features = test_features(downSampleIndex,:);
        
        train_location(testStartIndex:end,:) = [];
        train_features(testStartIndex:end,:) = [];
        
        nt = size(train_location,1);
        %  -- Assign validation set
        crossValid_index = 1:10:nt; % we choose 10 percent with jumping 10 step each time
        crossvalidation_features = train_features(crossValid_index,:);
        crossvalidation_location = train_location(crossValid_index,:);

        train_features(crossValid_index,:) = [];
        train_location(crossValid_index,:) = [];
    
end


%% -- plot training data set
handle_fig1 = figure('Name','locations:','NumberTitle','off');
axes1 = plot3(train_location(:,1),train_location(:,2),train_location(:,3),'MarkerFaceColor',[1 1 0],'MarkerEdgeColor',[1 0 1],...
    'Marker','o',...
    'LineWidth',2); % plot training sampling positions along with extracted features
% set(axes1,'ZTickLabel',{},'ZTick',zeros(1,0));
zlim([-1000 1000]);
zlim('manual')
xlabel('x direction (m)')
ylabel('y direction (m)')
hdt = datacursormode;
set(hdt,'DisplayStyle','window');
set(hdt,'UpdateFcn',{@labeldtips,train_features});


nf = size(train_features,2);

%% Plotting Map
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


%plotpredictedmaps(Predicted_Layer_Map,option);
%plotpredictedmaps(Prediction_Layer_Variance,option);

save('Layer-Map.mat','Predicted_Layer_Map','Prediction_Layer_Variance');
option.hyperparametrs = hyper_p;

%% Cross validation and feature selection
selected_nf = 1;
selected_featuters = (1:nf);
progressbar('Feature elemination ...')

[predicted.all, results(1),logL_all] = localizing( ...
    Predicted_Layer_Map,...                                % required Mapping mean
    Prediction_Layer_Variance,...                          % required Variance
    test_features,...                                           % measured feature
    test_location,...
    option);
IX = (1:16); % supposed to be 1:16, Mahdi tried different values
option_tmp = option;
option_tmp.hyperparametrs = option.hyperparametrs(:,IX);

[predicted.param4, results(2),~] = localizing( ...
    Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
    Prediction_Layer_Variance(:,:,IX),...                          % required Variance
    test_features(:,IX),...                                           % measured feature
    test_location,...
    option_tmp);
B_Predicted_Layer_Map = Predicted_Layer_Map;
B_Prediction_Layer_Variance = Prediction_Layer_Variance;
B_crossvalidation_features = crossvalidation_features;
B_test_features = test_features;
B_option = option;
for index1 = 1:(nf-selected_nf)
    if size(Predicted_Layer_Map,3)==16
        [predicted.CV16, results(3),~] = localizing( ...
            Predicted_Layer_Map,...                                % required Mapping mean
            Prediction_Layer_Variance,...                          % required Variance
            test_features,...                                           % measured feature
            test_location,...
            option);
    end
    [forward_rank, backward_rank] = ...
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
    [B,IY] = sort(selected_featuters);
    selected_featuters(IY(IX))= 1000 - index1;
    progressbar(index1/(nf-selected_nf))
end
[~,index1]=min(CX);
results(4) = nf - index1;
[~,B] = sort(selected_featuters);
IX = B(1:results(4));
B_option.hyperparametrs = B_option.hyperparametrs(:,IX);
[predicted.CVOpt, results(5),logL_opt] = localizing( ...
    B_Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
    B_Prediction_Layer_Variance(:,:,IX),...                          % required Variance
    B_test_features(:,IX),...                                           % measured feature
    test_location,...
    B_option);
% --- Calculate the BIC index for All and Selected cases ---
logLikelihood = - logLikelihood;
BIC_all = -2*sum(logLikelihood) + 3*size(train_features,2)*log(size(train_features,1));
BIC_selected = -2*sum(logLikelihood(IX)) + 3*results(4)*log(size(train_features,1));


disp(results)
disp('selected features')
disp(selected_featuters);
fprintf('BIC_all: %f',BIC_all);
fprintf('BIC_selected: %f',BIC_selected);
%close all
figure('name','prediction vs sample')
plot(test_location(:,1),test_location(:,2),'rd--','LineWidth',2);
hold on
%plot(predicted.all(:,1),predicted.all(:,2),'bo--','LineWidth',2);
%plot(predicted.CV16(:,1),predicted.CV16(:,2),'b*--','LineWidth',2);
plot(predicted.CVOpt(:,1),predicted.CVOpt(:,2),'b*--','LineWidth',2);
legend('test','prediction');
hold off


% nf = selected_nf;
% %% Estimate position of new samplin position using trained map
% [p0, RMS] = localizing( ...
%                      Predicted_Layer_Map,...                                % required Mapping mean
%                      Prediction_Layer_Variance,...                          % required Variance
%                      test_features,...                                           % measured feature
%                      test_location,...
%                      option);
%
% for index = 1: size(test_features,1)
%     p1 = p0(index,2);
%     p2 = p0(index,1);
%
% %     custom_sprintf(p1,p2,test_location(index,2),test_location(index,1));
%
%     if exist('handle_fig1')
%         figure(handle_fig1);
%         hold on;
%         pause(0.02)
%         plot3(p2,p1,test_location(index,3),...
%             'MarkerFaceColor',[0 1 0],...
%             'MarkerEdgeColor',[1 0 0],...
%             'Marker','o','markersize',10,...
%             'LineWidth',2) %plot estimates localization for the test image
%         plot3(  [p2, test_location(index,1)]',...
%                 [p1, test_location(index,2)]',...
%                 [test_location(index,3), test_location(index,3)]',...
%             'marker','.','color',[1 0 0]);
%     end
% end
% disp('===============================')
% disp(['RMS ERROR: ' num2str(RMS,2)])

toc
