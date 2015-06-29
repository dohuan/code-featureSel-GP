clear
clc
close all
%%
newData = 1; %0: load old data, 1: run new data

if (newData==1)
    features(1).name = 'ABS_FFT';
    features(1).data_type{1} = 'out_fix';
    features(1).nf = 64; % for one axis, actual nf = nf*2
    
    features(2).name = 'HISTOGRAM';
    features(2).data_type{1} = 'out_fix';
    features(2).nf = 64;
    
    features(3).name = 'WAVELET';
    features(3).data_type{1} = 'out_fix';
    features(3).nf = 72;
    
    outputCount = 1;
    for i=1:size(features,2)
        for j=1:size(features(i).data_type,2)
            features_temp.name = features(i).name;
            features_temp.nf = features(i).nf;
            features_temp.data_type = features(i).data_type{j};
            
            output_temp = main_function_movie(features_temp);
            
            output(outputCount) = output_temp;
            outputCount = outputCount + 1;
        end
    end
    nCase = size(output,2);
    save('./result/output_movie')
    clear *
end


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
%test_features = train_features(test_index,:);
test_location = location(test_index,:);
%crossvalidation_features = train_features(crossvalidation_index,:);
%crossvalidation_location = train_location(crossvalidation_index,:);
train_index = (1:dataSize)';

train_location = location;
train_location([test_index crossvalidation_index],:) = []; % remove test data from trainig dataset
train_index([test_index crossvalidation_index],1) = [];

n_train = size(train_location,1);
n_test  = size(test_location,1);

filename='grpLASSO_indoor_animated.avi';
vid = VideoWriter(filename);
vid.Quality = 100;
vid.FrameRate = 20;
open(vid)
frameRate = .05; % seconds between frames

load ./result/output_movie
for i=1:n_train
    s = [folderpath 'Panoramic_' num2str(train_index(i)) '.jpg'];
    img = imread(s);
    
    figure(1)
    movegui(figure(1),'northwest');
    % ----------
    subplot(4,4,[1 2 3 5 6 7 9 10 11])
    hold on
    if(i==1)
        plot(train_location(i,1),train_location(i,2),'k:','LineWidth',2.5);
    else
        plot([train_location(i-1,1) train_location(i,1)],...
             [train_location(i-1,2) train_location(i,2)],'k:','LineWidth',2.5);
    end
    hold off
    title('TRAIN PHASE')
    % ----------
    subplot(4,4,4)
    surf(output(1).GP_field_evol(:,:,i),'EdgeColor','none');
    title('FFT')
    % ----------
    subplot(4,4,8)
    surf(output(2).GP_field_evol(:,:,i),'EdgeColor','none');
    title('HIST')
    % ----------
    subplot(4,4,12)
    surf(output(3).GP_field_evol(:,:,i),'EdgeColor','none');
    title('SP')
    % ----------
    subplot(4,4,[13 14 15 16])
    imshow(img);
    title('Panoramic image') 
end
for i=1:n_test
    s = [folderpath 'Panoramic_' num2str(test_index(i)) '.jpg'];
    img = imread(s);
    
    figure(1)
    movegui(figure(1),'northwest');
    % ----------
    subplot(4,4,[1 2 3 5 6 7 9 10 11])
    hold on
    if(i==1)
        plot(output(1).BE_test(i,1),output(1).BE_test(i,2),'ro','LineWidth',2.5);
        plot(output(2).BE_test(i,1),output(2).BE_test(i,2),'bo','LineWidth',2.5);
        plot(output(3).BE_test(i,1),output(3).BE_test(i,2),'ko','LineWidth',2.5);
    else
        plot([output(1).BE_test(i-1,1) output(1).BE_test(i,1)],...
             [output(1).BE_test(i-1,2) output(1).BE_test(i,2)],'ro','LineWidth',2.5);
        plot([output(2).BE_test(i-1,1) output(2).BE_test(i,1)],...
             [output(2).BE_test(i-1,2) output(2).BE_test(i,2)],'bo','LineWidth',2.5);
        plot([output(3).BE_test(i-1,1) output(3).BE_test(i,1)],...
             [output(3).BE_test(i-1,2) output(3).BE_test(i,2)],'ko','LineWidth',2.5);
    end
    % ----------
    subplot(4,4,[13 14 15 16])
    imshow(img);
    title('Panoramic image') 
end

close(vid)