function [location, features] = extract_features_large(style,nf,mode,folderpath)
% if mode == 0
%     folderpath = '.\images2\';
% else
%     %folderpath = '.\imagesBoatRun\';                   % The folderpath of refrence file will be here
%     folderpath = '.\imagesAcuraRun\';
% end
load([folderpath 'location.mat']);                                          % load localization dataset
locationBuffer = location;
clear location
nt = size(locationBuffer,1);
str = [' feature extraction is started for ' num2str(nt) ' data set!'];
progressbar('Feature extraction ...')


%% Absolute FFT features
if strcmp(style,'ABS_FFT')
    disp(['FFT-ABS' str]);
    % --- take nf as number of features for EACH axis ---
    % --- so total feature = nf*2
    features = zeros(nt,nf*2);                                                 % we allocate memory for exteracted features
    for index=1:nt
        RGB_image = imread([folderpath num2str(index) '-d.jpg']);
        Gray_image = imresize(rgb2gray(RGB_image),[128,128]);
        F = fft2(Gray_image);
        % --- adding BOTH frequencies in x and y axes
        y = [reshape(reshape(abs(F(1,2:1+nf)),1,[]),[],1);...
             reshape(reshape(abs(F(2:1+nf,1)),1,[]),[],1)];
        features(index,:) = y;
        progressbar(index/nt)
    end
end

%% Absolute HISTOGRAM features
if strcmp(style,'HISTOGRAM')
    disp(['HISTOGRAM' str]);
    % --- take nf as number of features for EACH RGB channel ---
    % --- so total feature = nf*3
    %features = zeros(nt,nf*3);                                                 % we allocate memory for exteracted features
    for index=1:nt
        RGB_image = imread([folderpath num2str(index) '-d.jpg']);
        gray_image = rgb2gray(RGB_image);
        %y = imhist(gray_image,nf);
        y = [imhist(RGB_image(:,:,1),nf);...
             imhist(RGB_image(:,:,2),nf);...
             imhist(RGB_image(:,:,3),nf)];
        %features(index,:) = y(30:124);
        features(index,:) = y([13:64 13+64:64+64 13+2*64:64+64*2]);
        progressbar(index/nt)
    end
end

%% Wavelet features
if strcmp(style,'WAVELET')
    disp(['Wavelet' str]);
%     features = zeros(nt,78);                                                 % we allocate memory for exteracted features
%     for index=1:nt
%         RGB_image = imread([folderpath num2str(index) '-d.jpg']);
%         y = onlineimage(RGB_image);
%         features(index,:) = y;
%         progressbar(index/nt)
%     end
    features = zeros(nt,72);                                                 % we allocate memory for exteracted features
    for index=1:nt
        RGB_image = imread([folderpath num2str(index) '-d.jpg']);
        y = waveletExtractor(RGB_image,6);
        features(index,:) = y;
        progressbar(index/nt)
    end
end

%% sky fit
if strcmp(style,'SKYFIT')
    disp(['Skyfit' str]);
    features = zeros(nt,nf);                                                 % we allocate memory for exteracted features
    for index=1:nt
        file_name = sprintf('./imagesBoatRun/%d-d.jpg',index);
        img = imread(file_name);
        img_edge = edge(rgb2gray(img),'sobel');
        [ny,nx] = size(img_edge);
        lineProfile = zeros(1,nx);
        for i=1:nx
            col_temp = img_edge(:,i);
            IX = find(col_temp==1);
            if isempty(IX)==1
                if i==1
                    lineProfile(i,1) = round(ny/2);
                else
                    lineProfile(1,i) = lineProfile(1,i-1);
                end
            else
                lineProfile(1,i) = round(mean(IX));
            end
        end
        f = fft(lineProfile);
        y = reshape(reshape(abs(f(1,2:1+nf)),1,[]),[],1);
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
    %hyper_p = zeros(4,nf);
    progressbar('Hyperparameters estimation ...')
    if ((mode == 1)||(mode == 2))
        % ---- convert to meter coordinate, origin at min([Long Lat]) -----
        dataSize = size(locationBuffer,1);
        origin_m = [min(locationBuffer(:,1)) min(locationBuffer(:,2))]; %gps coor of the origin of meter coor
        for i=1:dataSize
            % -- lldistkm([lat1 long1],[lat2 long2])
            [Long_m_tmp,~] = lldistkm([origin_m(1,2) locationBuffer(i,1)],...
                                      [origin_m(2) origin_m(1)]);
            [Lat_m_tmp,~]  = lldistkm([locationBuffer(i,2) origin_m(1,1)],...
                                      [origin_m(2) origin_m(1)]);
            location(i,:)  = 1000*[Long_m_tmp Lat_m_tmp];
        end
        location = [location,(1:dataSize)'];
    else
        dataSize = size(locationBuffer,1);
        location = [locationBuffer,(1:dataSize)'];
    end
%     log_likelihood = zeros(nf,1);
%     for index = 1:nf
%         p0 = [var(features(:,index)) 5 5 0.01];
%         [hyper_p(:,index), log_likelihood(index)] = HyperParameter(features(:,index),location(:,1:2)); %extract hyper parameters for each layer sepratly
%         %[hyper_p(:,index), ~] = HyperParameter(features(:,index),location(:,1:2),p0); %extract hyper parameters for each layer sepratly
%         %hyper_p(:,index) = [1;0.5;0.5;0.01];
%         str = sprintf('hyper-parameters # %d: \t (sig_f^2: %0.2f, \t sig_x: %0.2f, \t sig_y: %0.2f, \t sig_w^2: %0.2f)',index,hyper_p(1,index),hyper_p(2,index),hyper_p(3,index),hyper_p(4,index));
%         disp(str)
%         progressbar(index/nf)
%         %log_likelihood(index) = log_likelihood + log_temp;
%     end
    
    %save('previously_computed_features.mat','location', 'features','hyper_p')
end