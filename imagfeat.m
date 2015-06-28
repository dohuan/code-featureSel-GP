%%%%%   This code calculates the eigenvector basis from the environment

%%% IMPORTANT VARIABLES %%%

% 'imagefolder' -> Please put all the images from environment in one folder and give its path
% to the 'imagefolder' variable, you would also need to modify full path name in taking input 
% for variable 'currentimage'


% 'environmat' -> this contains all the image information(observation) as rows
% and with columns representing features. Notice that in the given example, 
% environmat is going to have size 35*126 where 35 represents 35 images in
% the given folder (data by Dr Lee). The complete break up of 126 is given
% below: (you can ignore that if you want to)
%           Given Image broken into two sub-images
%           Each sub-image has 60 features from dtcwt for 5 scales
%                   60 is broken down as 12 for each scale
%                        12 is broken down as 6 real and 6 complex filters
%                           6 here is the average filter response in 6
%                           different orientations
%           3 more features (norm red norm blue norm green) are appended to
%           60 
%  Total break up becomes = 2*(5*(6*2)+3)=126

%'varcaptured'-> this is important and gives info of how much variance is
%captured. You can display this variable and look at its first 10-20
%values. If varcaptured(5) = 0.87, it means that first five eigenvectors
%would capture 87% variability. In this case, 10 are capturing 97% variance

%'selectedbasis'-> this is the final output, the value 'n' in
%selectedbasis(:,1:n) can be changed to capture more or less
% variability as gauged by varcaptured. Like 5 in place of n would 
% mean capture 87% variability and so on

%%%%%

% the variable 'selectedbasis' is stored and used in onlineimage program
% you might have to modify the location of the saved file so that it is
% available to 'onlineimage.m'


clear all
clc
q=1;
% path = 'C:\Users\jadaliha\Dropbox\code\robotics vision\';
path = '';
imagefolder=dir(fullfile(path,'images','*.jpg'));  %The path of images folder here

for i=1:length(imagefolder)
    currentimage=imread(fullfile(path,'images',imagefolder(i).name));
    fullfile(path,'images',imagefolder(i).name)
dumq=1;

%dd=imread('E:\robotics vision\images\1-c.jpg');

imorig = imresize(currentimage, [256 256]);
imagquotient=1;

%% Calculation of normalized red,green blue
for rimage=1:2
    B=imorig(imagquotient:128+imagquotient-1,:,:);
imagquotient=imagquotient+128;
red=B(:,:,1);
red=im2double(red);
green=B(:,:,2);
green=im2double(green);
blue=B(:,:,3);
blue=im2double(blue);

for i=1:128
    for j=1:256
        div=sqrt(red(i,j)^2 + green(i,j)^2 + blue(i,j)^2);
        if(div==0)
            div=0.5774;
        end
        red(i,j)=red(i,j)/div;
        green(i,j)=green(i,j)/div;
        blue(i,j)=blue(i,j)/div;
    end
end

avred=mean(mean(red));
avgreen=mean(mean(green));
avblue=mean(mean(blue));
avvector=[avred avgreen avblue];

%%%%%%%%%%%%%%%%%%

%calculation of dtcwt coefficients 

tic
n=rgb2gray(B);
nn=im2double(n);
nn=nn*255;
  
    
    J = 5;
    [Faf, Fsf] = FSfarras;
    [af, sf] = dualfilt1;
    w = cplxdual2D(nn, J, Faf, af);            % w(j)(r)(k)(d)
    
    stupvar=1;
for jj=1:J
    for par=1:2
        for orik=1:2
            for orid=1:3
                if(stupvar==1)
             dummat=reshape([w{jj}{par}{orik}{orid}]',1,[]);
             dummat=mean(dummat);
                else
            resmat=mean(reshape([w{jj}{par}{orik}{orid}]',1,[]));
            dummat=[dummat resmat];
            
                end
              
            stupvar=2;
            end
        end

    end
end
dummat=[dummat avvector];

if(dumq==1)
    newdum=dummat;
else
    newdum=[newdum dummat];
end
dumq=2;

end

if(q==1)
environmat=newdum;

else
    environmat=[environmat;newdum];     %environmat contains all the feature vectors 

end

q=2;
toc 
end
menv=mean(environmat);
for i=1:length(environmat)
    environmat(:,i)=environmat(:,i)-menv(i);
end
% [basis,evs1]=eig(cov(environmat));
% eigens=diag(evs1);

%environmat=zscore(environmat);

%% pca on obtained featurevector
[basis,imagfet,evs]=princomp(environmat);

varcaptured=cumsum(evs)./sum(evs); %varcaptured tells about the percentage variance captured
selectedbasis=basis(:,1:8);       %value 1:n here would mean that select first 10 eigenvectors for basis
save([path 'selectedbasis.mat'],'selectedbasis');