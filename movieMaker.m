clear
clc
close all
%%

folderpath = './data/imagesAcuraRun/merged(K_L_fixedAngle)/';
load([folderpath 'location.mat']);
locationBuffer = location;
clear location
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
crossvalidation_index = 1:5:dataSize; % we choose 10 percent with jumping 10 step each time
test_index = crossvalidation_index + 1; % we keep 10% cross validation data
test_features = train_features(test_index,:);
test_location = train_location(test_index,:);
crossvalidation_features = train_features(crossvalidation_index,:);
crossvalidation_location = train_location(crossvalidation_index,:);

train_location = location;
train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset

filename='grpLASSO_indoor_animated.avi';
vid = VideoWriter(filename);
vid.Quality = 100;
vid.FrameRate = 20;
open(vid)
frameRate = .05; % seconds between frames

load ./result/ABS_FFT-out_fix-data Predicted_Layer_Map


close(vid)