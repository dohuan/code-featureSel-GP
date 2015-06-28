clear
clc
close all
tic
addpath(genpath('./matlabPyrTools/'))

feature_name = 'ABS_FFT';
lucky_number = 100;
test_image = imread(['.\imagesAcuraRun\merged(K_L)\' num2str(lucky_number)...
    '-d.jpg']);
load '.\imagesAcuraRun\merged(K_L)\location.mat'
test_location = location(lucky_number,1:2);
output_1 = localize_me_app(feature_name,test_image,test_location);

feature_name = 'HISTOGRAM';
lucky_number = 100;
test_image = imread(['.\imagesAcuraRun\merged(K_L)\' num2str(lucky_number)...
    '-d.jpg']);
load '.\imagesAcuraRun\merged(K_L)\location.mat'
test_location = location(lucky_number,1:2);
output_2 = localize_me_app(feature_name,test_image,test_location);

feature_name = 'WAVELET';
lucky_number = 100;
test_image = imread(['.\imagesAcuraRun\merged(K_L)\' num2str(lucky_number)...
    '-d.jpg']);
load '.\imagesAcuraRun\merged(K_L)\location.mat'
test_location = location(lucky_number,1:2);
output_3 = localize_me_app(feature_name,test_image,test_location);

save computation_time