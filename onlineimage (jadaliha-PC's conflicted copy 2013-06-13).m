function [featurevalue] = onlineimage(filename)
% path = 'C:\Users\jadaliha\Dropbox\code\robotics vision\';
path = '';
load([path 'selectedbasis.mat']);

% currentimage=imread([path 'images\01-c.jpg']);
currentimage=imread(filename);
% imagefolder=dir(fullfile('e:','robotics vision','images','*.jpg'));  
% newcor=1;
% for i=1:length(imagefolder)
%     currentimage=imread(fullfile('e:','robotics vision','images',imagefolder(i).name));
    %fullfile('e:','robotics vision','images',imagefolder(i).name)

dumq=1;
imorig = imresize(currentimage, [256 256]);
tic

imagquotient=1;
for i=1:2
B=imorig(imagquotient:128+imagquotient-1,:,:);
red=B(:,:,1);
red=im2double(red);
green=B(:,:,2);
green=im2double(green);
blue=B(:,:,3);
blue=im2double(blue);
imagquotient=imagquotient+64;
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
if(dumq==1)
    newmat=dummat;
else
    newmat=[newmat dummat];
end
dumq=2;
newmat=[newmat avvector];
end
newmat=newmat-mean(newmat);

featurevalue=newmat*selectedbasis;
 
