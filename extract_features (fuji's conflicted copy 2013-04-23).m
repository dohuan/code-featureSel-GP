function [location, features, hyper_p] = extract_features(style,nf)
path = 'h:\Dropbox\GMRF-SLAM\Simulations\';                                 % The path of refrence file will be here
load('location.mat');                                                       % load localization dataset
nt = size(location,1);
str = [' feature extraction is started for ' num2str(nt) ' data set!'];
progressbar('Feature extraction ...')

%% Absolute FFT features
if strcmp(style,'ABS_FFT')
    disp(['FFT-ABS' str]);
    features = zeros(nt,nf);                                                 % we allocate memory for exteracted features
    for index=1:nt 
        RGB_image = imread(['images/' num2str(index) '-d.jpg']);
        Gray_image = imresize(rgb2gray(RGB_image),[128,128]);
        F = fft2(Gray_image);
        y = reshape(reshape(abs(F(1,2:1+nf)),1,[]),[],1);
        features(index,:) = y;
        progressbar(index/nt)
    end 
end

%% Wavelet features
if strcmp(style,'WAVELET')
    disp(['Wavelet' str]);
    features = zeros(nt,8);                                                 % we allocate memory for exteracted features
    for index=1:nt 
        y = onlineimage(['images/' num2str(index) '-d.jpg']);
        features(index,:) = y;
        progressbar(index/nt)        
    end 
end


%% Load previously computed features
if strcmp(style,'LOAD')
    load('previously_computed_features.mat')
    progressbar(1)
else
    %% we normalize features here
    features = features - ones(size(features,1),1)*mean(features,1); % remove mean average
    cov_features = cov(features);
    [U,S,~] =  svd(cov_features); % make feature orthogonal using SVD 
    features = features*U*sqrt(S^-1); %transfer top new coordinates
    
    %% Predict hyper parameters
    nf = size(features,2);
    hyper_p = zeros(3,nf);
    progressbar('Hyperparameters estimation ...')
    for index = 1:nf
        hyper_p(:,index) = HyperParameter(features(:,index),location); %extract hyper parameters for each layer sepratly
        % hyper_p(:,index) = [1;0.5;0.5]; 
        disp(['hyper-parameters #', num2str(index), ' :']); % hyper parameters for the #index feature
        disp(['sig_f^2:'  ,num2str(hyper_p(1,index)), ...
            '      sig_x:',num2str(hyper_p(2,index)),...
            '      sig_y:',num2str(hyper_p(3,index))]);
        progressbar(index/nf)
    end

    save('previously_computed_features.mat','location', 'features','hyper_p')
end