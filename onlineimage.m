function [feature] = onlineimage(currentimage)

featurevalue = zeros(78,1);
imorig = imresize(currentimage, [192 256]);
J = 4;
[Faf, ~] = FSfarras;
[af, ~] = dualfilt1;

for level=1:3
    B=imorig(level*64-63:level*64,:,:);
    Er = mean(mean(B(:,:,1)));
    Eg = mean(mean(B(:,:,2)));    
    Ergb = mean(mean(B(:,:,1)))+Er+Eg;
    featurevalue(level*26-25) = (Er/Ergb);
    featurevalue(level*26-24) = (Eg/Ergb);
    n=rgb2gray(B);
    nn=im2double(n)*255;
    w = cplxdual2D(nn, J, Faf, af);            % w(j)(r)(k)(d)
%   w{j}{i}{d1}{d2} - wavelet coefficients
%       j = 1..J (scale)
%       i = 1 (real part); i = 2 (imag part)
%       d1 = 1,2; d2 = 1,2,3 (orientations)
    for index1=1:4                  % scale
        for index2=1:2          % direction
            for index3=1:3      % direction
                index0 = level*26+index1*6+index2*3+index3 - 33;
                featurevalue(index0) = ...
                    mean(mean(w{index1}{1}{index2}{index3}));
            end
        end
    end
end

feature = featurevalue';