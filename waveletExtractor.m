function feature = waveletExtractor(inImg,level)
%% --------------- Extract Steerable Pyramid from input image -------------
%                           Author: Huan N. Do
% Prequisite: steerable pyramid MATLAB package
% - Input:
%       + inImg: Gray-scale image with squared nxn size
%       + level: Number of pyr levels
% - Output:
%       + feature: 72-D feature vector, described in detail in Brook2008
%       paper
feature = [];
img = rgb2gray(imresize(inImg,[800 800]));
imgSize = size(img,1);

[pyr,pind] = buildSpyr(img,level,'sp3Filters');
dropIndex = 1:imgSize*imgSize;
% -- drop first image in pyr
pyr(dropIndex) = [];
for i=1:level
    pyrIndex = imgSize/2^(i-1);
    dropIndex = 1:pyrIndex*pyrIndex*4;
    subBand = pyr(dropIndex); % actually a group of 4 sub-bands
    pyr(dropIndex) = [];
    %subBand = reshape(subBand,[],pyrIndex);
    sizeBand = size(subBand,1);
    split = (round(sizeBand*(1:11)/12))';
    for j=1:size(split,1)
        if (j==1)
            feature = [feature,mean(subBand(1:split(j)))];
        else
            feature = [feature,mean(subBand(split(j-1)+1:split(j)))];
        end
    end
    feature = [feature,mean(subBand(split(j)+1:end))];
end

end