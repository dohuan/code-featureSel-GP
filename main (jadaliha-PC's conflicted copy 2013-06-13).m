clear
clc
close all

option.x_limit = [1 10];     % The limit on the x cordinate will be set here
option.y_limit = [1 26];
option.resolution = 0.2;    % The resulotion of predicted map will be determined here
global system
system.sigma2w = 0.01;       % The measurements noise will be set here
%% Load data and doing pre processings
[location, features] = extract_features('WAVELET',8);     % Choices are {ABS_FFT, WAVELET, LOAD}
nf = size(features,2); % number of features used.
%% we normalize features here
features = features - ones(size(features,1),1)*mean(features,1); % remove mean average
cov_features = cov(features);
[U,S,V] =  svd(cov_features); % make feature orthogonal using SVD 
features = features*U*sqrt(S^-1); %transfer top new coordinates
nt = size(location,1);

%% seperate test set
test_index = 1:10:nt; % we choose 10 percent with jumping 10 step each time
test_features = features(test_index,:);
test_location = location(test_index,:);
features(test_index,:) = []; % remove test data from trainig dataset
location(test_index,:) = []; % remove test data from trainig dataset

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


for index =1:nf; 
    
%% Predict hyper parameters
hyper_p(:,index) = HyperParameter(features(:,index),location); %extract hyper parameters for each layer sepratly
% hyper_p(:,index) = [1;2;2]; 
disp(['hyper-parameters #', num2str(index), ' :']); % hyper parameters for the #index feature
disp(['sig_f^2:'  ,num2str(hyper_p(1,index)), ...
    '      sig_x:',num2str(hyper_p(2,index)),...
    '      sig_y:',num2str(hyper_p(3,index))]);

%% Plotting Map
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

tmp1 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp2 = (option.y_limit(1):option.resolution:option.y_limit(2));
sigma2xstar = diag(sigma2xstar);
zhatxstar = reshape(zhatxstar,size(tmp2,2),size(tmp1,2),[]);
sigma2xstar = reshape(sigma2xstar,size(tmp2,2),size(tmp1,2),[]);
figNaive = figure('Name',['Feauture #', num2str(index), ' :'],'Position',[100 50 400 670],'NumberTitle','off');
axes('Parent',figNaive,'Position',[0.11 0.55 0.85 0.37]);
imagesc(tmp1 ,tmp2,reshape(zhatxstar,size(tmp2,2),size(tmp1,2)),'DisplayName','Naive Conditional Mean')
title('Predicted field');
colorbar

axes('Parent',figNaive,'Position',[0.11 0.05 0.85 0.37]);         
imagesc(tmp1 ,tmp2,sigma2xstar,'DisplayName','Naive Conditional Variance')        
hold on
tmp = X;
plot(tmp(:,1)',tmp(:,2)','Color',[1 0 1],'MarkerSize',10,'Marker','x'...
    ,'LineWidth',3,'LineStyle','none','DisplayName','Real Position');    
title('Prediction variance');
colorbar

Predicted_Layer_Map(:,:,index) = zhatxstar;
Prediction_Layer_Variance(:,:,index) = sigma2xstar;

end

save('Layer-Map.mat','Predicted_Layer_Map','Prediction_Layer_Variance');




%% Estimate position of new samplin position using trained map
str = 'Estimated position = ( %2.1f \t %2.1f)  vs True position = ( %2.1f \t %2.1f)';

map_dimension = size(Predicted_Layer_Map); 
testpoint_features = zeros(map_dimension);
error_square = 0;
for index = 1: size(test_features,1)
    for i=1:nf
        testpoint_features(:,:,i) = ones(map_dimension(1:2))*test_features(index,i);
    end
    cost_function = (testpoint_features - Predicted_Layer_Map); % "Predicted_Layer_Map" the trained data map is used here
    summedup_cost_function = sum(cost_function.^2,3); % find the a location which has most similar predicted features to the extracted feature for the current test image

    [C1,I1] = min(summedup_cost_function);
    [~, I2] = min(C1);

    tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
    tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
    p1 = tmp1(I1(I2));
    p2 = tmp2(I2);
    disp(sprintf(str,p1,p2,test_location(index,2),test_location(index,1)));

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