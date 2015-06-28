% The current extension mode is zero-padding (see dwtmode).
clear all
% close all
% Load image. 
load tire 
% X contains the loaded image.

% For an image the decomposition is performed using: 
t = wpdec2(X,2,'db1'); 
% The default entropy is shannon.

% Plot wavelet packet tree 
% (quarternary tree, or tree of order 4). 
figure; plot(t)


figure;image(wprec2(t));