clear
clc
close all

option.x_limit = [1 10];     % The limit on the x cordinate will be set here
option.y_limit = [1 26];
option.resolution = 0.2;    % The resulotion of predicted map will be determined here
global system
system.sigma2w = 0.01;       % The measurements noise will be set here
%% Load data and doing pre processings
[location, features, hyper_p] = extract_features('WAVELET',8);     % Choices are {ABS_FFT, WAVELET, LOAD}

nt = size(location,1);

%% seperate test set
test_index = 1:10:nt; % we choose 10 percent with jumping 10 step each time
crossvalidation_index = test_index + 1; % we keep 10% cross validation data
test_features = features(test_index,:);
test_location = location(test_index,:);
crossvalidation_features = features(test_index,:);
crossvalidation_location = location(test_index,:);

features([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset

%% plot training data set
handle_fig1 = figure('Name','locations:','NumberTitle','off');
axes1 = plot3(location(:,1),location(:,2),location(:,3),'MarkerFaceColor',[1 1 0],'MarkerEdgeColor',[1 0 1],...
    'Marker','o',...
    'LineWidth',2); % plot training sampling positions along with extracted features
% set(axes1,'ZTickLabel',{},'ZTick',zeros(1,0));
zlim([-1000 1000]);
zlim('manual')
xlabel('x direction (m)')
ylabel('y direction (m)')
hdt = datacursormode;
set(hdt,'DisplayStyle','window');
set(hdt,'UpdateFcn',{@labeldtips,features});


nf = size(features,2);

Feature_Selection = [];
if ~isempty(Feature_Selection)
    if (Feature_Selection == 1);
        % Feature Selection using Pearson Correlation Coefficient and ...
        nf = 8;                             %size(features,2); % number of features used.
        R = corrcoef([features,location]);  % Compute correlation between features (x) and output (y)
        PearsonCorrCoef = R(1:size(features,2),size(features,2)+1:end-1);      

        handle_fig2 = figure('Name','Pearson Correlation Coefficient','NumberTitle','off');

        PearsonCorrCoef = PearsonCorrCoef * diag(sum(abs(PearsonCorrCoef)).^(-1));  % Normalization over output dimension
        bar(PearsonCorrCoef);

        [B,IX] = sort(sum(abs(PearsonCorrCoef),2)); % feature information sorting
        IX = IX(end-nf+1:end);
        features = features(:,IX);
        test_features = test_features(:,IX);
    else
        
    end
end

%% Plotting Map
tmp1 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp2 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng1 = size(tmp1,2);
ng2 = size(tmp2,2);
Predicted_Layer_Map = zeros(ng2,ng1,nf);
Prediction_Layer_Variance = zeros(ng2,ng1,nf);

for index =1:nf; 
% Produce grid points
[x,y] = meshgrid(option.x_limit(1):option.resolution:option.x_limit(2),option.y_limit(1):option.resolution:option.y_limit(2));
xstar = [reshape(x,[],1),reshape(y,[],1)];
X = location(:,1:2);
Y = features(:,index);
 
sigma2w = system.sigma2w;
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
zhatxstar = reshape(zhatxstar,size(tmp2,2),size(tmp1,2),[]);
sigma2xstar = reshape(sigma2xstar,size(tmp2,2),size(tmp1,2),[]);

Predicted_Layer_Map(:,:,index) = zhatxstar;
Prediction_Layer_Variance(:,:,index) = sigma2xstar;
end

% plotpredictedmaps(Predicted_Layer_Map,option)

save('Layer-Map.mat','Predicted_Layer_Map','Prediction_Layer_Variance');

%% Cross validation and feature selection
[forward_rank, backward_rank] = ...
    feature_ranking( Predicted_Layer_Map,  ...
                     crossvalidation_features, ...
                     crossvalidation_location, option);






%% Estimate position of new samplin position using trained map
map_dimension = size(Predicted_Layer_Map); 
testpoint_features = zeros(map_dimension);
error_square = 0;
tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
for index = 1: size(test_features,1)
    for i=1:nf
        testpoint_features(:,:,i) = ones(map_dimension(1:2))*test_features(index,i);
    end
    cost_function = (testpoint_features - Predicted_Layer_Map); % "Predicted_Layer_Map" the trained data map is used here
    summedup_cost_function = sum(cost_function.^2,3); % find the a location which has most similar predicted features to the extracted feature for the current test image

    [C1,I1] = min(summedup_cost_function);
    [~, I2] = min(C1);


    p1 = tmp1(I1(I2));
    p2 = tmp2(I2);
    
    custom_sprintf(p1,p2,test_location(index,2),test_location(index,1));

    if exist('handle_fig1')
        figure(handle_fig1);
        hold on;
        pause(0.02)
        plot3(p2,p1,test_location(index,3),...
            'MarkerFaceColor',[0 1 0],...
            'MarkerEdgeColor',[1 0 0],...
            'Marker','o','markersize',10,...
            'LineWidth',2) %plot estimates localization for the test image
        plot3(  [p2, test_location(index,1)]',...
                [p1, test_location(index,2)]',...
                [test_location(index,3), test_location(index,3)]',...
            'marker','.','color',[1 0 0]);
        error_square =  error_square + ...
                        (p2 - test_location(index,1))^2 + ...
                        (p1 - test_location(index,2))^2;
    end
end
disp('===============================')
RMS = sqrt(error_square/size(test_features,1));
disp(['RMS ERROR: ' num2str(RMS,2)])


% 
% %% plot a prety feature map
% img = load('clown');
% I = repmat(img.X,[1 1 5]);
% cmap = img.map;
% 
% %# coordinates
% [X,Y] = meshgrid(1:size(I,2), 1:size(I,1));
% Z = ones(size(I,1),size(I,2));
% 
% %# plot each slice as a texture-mapped surface (stacked along the Z-dimension)
% for k=1:size(I,3)
%     surface('XData',X-0.5, 'YData',Y-0.5, 'ZData',Z.*4*k, ...
%         'CData',I(:,:,k), 'CDataMapping','direct', ...
%         'EdgeColor','none', 'FaceColor','texturemap')
% end
% colormap(cmap)
% view(3), box on, axis tight square
% set(gca, 'YDir','reverse', 'ZLim',[0 size(I,3)+1])