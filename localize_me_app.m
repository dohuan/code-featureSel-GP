function [predicted_location, collapsedtime] = ...
                     localize_me_app(features_data,test_image,test_location)
%% Return the localization time for one test point, given an image

% --- load the map ---
filename = ['./main_f/' features_data.name '-' features_data.data_type '-data.mat'];
load(filename);
nf = features_data.nf;
tic
if strcmp(features_data.name,'ABS_FFT')
    Gray_image = imresize(rgb2gray(test_image),[128,128]);
    F = fft2(Gray_image);
    test_features = [reshape(reshape(abs(F(1,2:1+nf)),1,[]),[],1);...
             reshape(reshape(abs(F(2:1+nf,1)),1,[]),[],1)];
end
if strcmp(features_data.name,'HISTOGRAM')
    %Gray_image = rgb2gray(test_image);
    %test_features = imhist(Gray_image,nf);
    y =                [ imhist(test_image(:,:,1),nf);...
                         imhist(test_image(:,:,2),nf);...
                         imhist(test_image(:,:,3),nf)];
    %y = imhist(rgb2gray(test_image),nf);
    test_features = y([13:64 13+64:64+64 13+2*64:64+64*2]);
    
end
if strcmp(features_data.name,'WAVELET')
    test_features = waveletExtractor(test_image,6);
end

test_features = reshape(test_features,1,[]);
option.hyperparametrs = option.hyperparametrs(:,IX);
[predicted_location, ~,~] = localizing( ...
    Predicted_Layer_Map(:,:,IX),...                                % required Mapping mean
    Prediction_Layer_Variance(:,:,IX),...                          % required Variance
    test_features(IX),...                                           % measured feature
    test_location,...
    option);

collapsedtime = toc;
fprintf('%s takes %f seconds for one localization\n',features_data.name,collapsedtime)
end