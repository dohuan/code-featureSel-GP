imagefolder=dir(fullfile('e:','robotics vision','images','*.jpg'));  
for i=1:length(imagefolder)
    currentimage=imread(fullfile('e:','robotics vision','images',imagefolder(i).name));
    imshow(currentimage);
end   
%dd=imread('E:\robotics vision\images\1-c.jpg');
%n=rgb2gray(dd);
%nn=im2double(n);
