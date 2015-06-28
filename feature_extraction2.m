path = 'C:\Users\jadaliha\Dropbox\';
load([path 'GMRF-SLAM\Simulations\12-9-2012-dataset_abs.mat']);
load([path 'Data sets\panoramic image\data\12-9-2012-true_position.mat'])
load([path 'Data sets\panoramic image\data\12-9-2012-background.mat'])
load([path 'Data sets\panoramic image\data\12-9-2012-localization.mat']);    % motion tracking results
load([path 'Data sets\panoramic image\data\12-9-2012-grid-rulers.mat'])


h_background = imshow(background);
for i = 1:10
    for j = 1:26
        grids((i-1)*26+j,:) = grid_points(i,:,j);
    end
end
hold on

h_grids = plot(grids(:,1),grids(:,2),...
                'marker','o',           'linestyle','none',...
                'color',[1 0.75 0.75],    'linewidth',2,'markersize',2,...
                'DisplayName','Griding system');

            
% Fetch a new location and compute possible sampling positions according 
% the grid positions.
h_possiblity = plot(grids(:,1),grids(:,2),... % plot possible sampling positions
                'marker','o',           'linestyle','none',...
                'color',[0 0.75 0],    'linewidth',2,'markersize',5,...
                'DisplayName','Possible positions');
            
            
h_measured = plot(0,0,... %plot measured position
                'marker','*',           'linestyle','none',...
                'color',[0 0.2 0],    'linewidth',2,'markersize',8,...
                'DisplayName','Measured positions'); 
            

h_true= plot(0,0,... %plot true position, manually collected
                'marker','square',           'linestyle','none',...
                'color',[0 0.2 0.3],    'linewidth',8,'markersize',8,...
                'DisplayName','True positions'); 
            

Y = [];
globaloption.numberoffields = 8;
sys_parameter.torus = 0;
for t = 1:825
    x_ = 1920/1489*true_position(t,1);
    y_ = 1080/816*true_position(t,2);
    distance2 = (grids(:,1)-x_).^2 + (grids(:,2)-y_).^2;
    [~,IX] = sort(distance2);
    x_true = grids(IX(1),1);
    y_true = grids(IX(1),2);
    set(h_true, 'XData',x_ , 'YData', y_);
    AgumentedData(1,t).possible_q.true_q = IX(1);
    
    x_ = localization(t,2);
    y_ = localization(t,3);
    set(h_measured, 'XData', x_, 'YData', y_);

    numberofpossib = 4 + floor((1080-y_)/250); % number of possible sampling positions for each measurted point

    distance2 = (grids(:,1)-x_).^2 + (grids(:,2)-y_).^2;
    [B,IX] = sort(distance2);
    IX = IX(1:numberofpossib);
    x_possible = grids(IX,1);
    y_possible = grids(IX,2);
    AgumentedData(1,t).possible_q.measuredposition = IX(1);
    AgumentedData(1,t).possible_q.support_qt = mat2cell(IX);
    AgumentedData(1,t).possible_q.prior_qt = 1/numberofpossib;
    AgumentedData(1,t).possible_q.N_possible_qt = numberofpossib;
    
    set(h_possiblity, 'XData',x_possible , 'YData', y_possible);
    

    
%     RGB_image = imread(['images3/' num2str(t) '-b.jpg']);
%     Gray_image = imresize(rgb2gray(RGB_image),[128,128]);
%     F = fft2(Gray_image,16,16);
%     tmp = reshape([reshape(real(F),1,[]);reshape(imag(F),1,[])],[],1);
%     tmp = reshape(reshape(abs(F),1,[]),[],1);
    tmp = reshape(onlineimage([path 'GMRF-SLAM\Simulations\images3\' num2str(t) '-d.jpg']),[],1);
    Y = [Y,tmp];
    

    

    
    pause(0.1)
    
end

% [COEFF,SCORE,latent,tsquare] = princomp(Y','econ') ; % SCORE = Y' * COEFF - E(Y')
% tmp = cumsum(latent)./sum(latent);
% disp(tmp(1:globaloption.numberoffields))
% Y = SCORE(:,1:globaloption.numberoffields)'; %check this equation

muY = mean(Y,2);
eig(Y*Y')
muz_theta = kron(muY,ones(260,1));
globaloption.Gam = cov(Y');

for t = 1:825
    AgumentedData(1,t).y = Y(:,t);
end
globaloption.T_mesh=(1:825);
globaloption.vehicle.ObservationNoise = 1000;
globaloption.vehicle.ModelUncertanity = 2;
save([path 'GMRF-SLAM\Simulations\12-9-2012-dataset_wave.mat'],...
    'globaloption','sys_parameter','AgumentedData','muz_theta');
