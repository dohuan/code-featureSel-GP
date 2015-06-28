%      Gaussian Process Dynamic Model (GPDM) for correcting RMSE
%%
% Taking the predicted trajectory then apply the GPDM
% Notation consistent with the write-up:
% "Gaussian Process model for correcting RMSE" by Huan Do.

clear
clc
close all
tic
addpath(genpath('./gpml-matlab/'))    % GPML toolbox
% --- Config GPML ---
covfunc = @covSEard;
likfunc = @likGauss;

L = 3; % Lag horizon
load_mat = matfile('main_f/output100714(final).mat');
output = load_mat.output;
load train_location.mat
% output:
% 1- ABS_FFT-64-in
% 2- ABS_FFT-64-out_unfix
% 3- ABS_FFT-64-out_fix
% 4- HISTOGRAM-64-in
% 5- HISTOGRAM-64-out_unfix
% 6- HISTOGRAM-64-out_fix
% 7- WAVELET-72-in
% 8- WAVELET-72-out_unfix
% 9- WAVELET-72-out_fix
output_index = 2; % Choose the feature to work on

% --- center the path by substract the middle of the field ---
nt_train = size(train_location,1);
mu_x = (max(train_location(:,1))+min(train_location(:,1)))/2;
mu_y = (max(train_location(:,2))+min(train_location(:,2)))/2;
train_location = [train_location(:,1)-ones(nt_train,1)*mu_x,...
    train_location(:,2)-ones(nt_train,1)*mu_y];

test_location       = output(output_index).BE_test;
true_test_location  = output(output_index).test(:,1:2);

% true_test_location = train_location(201:end,1:2);
% train_location(201:end,:) = [];
% nt_train = size(train_location,1); % update train size

nt_test = size(true_test_location,1);

% --- try with test set taken from train ---
% noiseStd = 5;
% noise_X = noiseStd*randn(nt_test,1);
% noise_Y = noiseStd*randn(nt_test,1);
% test_location = true_test_location + [noise_X noise_Y];

% --- center the path by substract the middle of the field ---
test_location = [test_location(:,1)-ones(nt_test,1)*mu_x,...
    test_location(:,2)-ones(nt_test,1)*mu_y];
true_test_location = [true_test_location(:,1)-ones(nt_test,1)*mu_x,...
    true_test_location(:,2)-ones(nt_test,1)*mu_y];

% --- Calculate mean and variance of Root-Squared-Error (RSE) ---
rse_temp = sqrt((test_location-true_test_location).^2);
rse.mean = mean(rse_temp);
rse.var  = var(rse_temp);
%rse.var  = [noiseStd noiseStd].^2;


%%            Learning hyper-parameter from train location
% --- create predictor matrix with L-horizon to the past ---
count = 1;
for i=L+1:nt_train
    train_features(count,:,1) = train_location(i-1:-1:i-L,1)';
    train_features(count,:,2) = train_location(i-1:-1:i-L,2)';
    count = count + 1;
end
% -- cut first L locations --
train_location(1:L,:) = [];
nt_train = size(train_location,1); % update size of train set

% --- Learn hyper-parameter ---
% hyp(1) for x-aixs, hyp(2) for y-axis
hyp(1).cov = log([ones(L,1); var(train_location(:,1))]);
hyp(2).cov = log([ones(L,1); var(train_location(:,2))]);
hyp(1).lik = log(rse.var(1)); % put variance of RSE here
hyp(2).lik = log(rse.var(2)); % put variance of RSE here

hyp(1) = minimize(hyp(1), @gp, -500, @infExact, [], covfunc, likfunc,...
    train_features(:,:,1), train_location(:,1));
hyp(2) = minimize(hyp(2), @gp, -500, @infExact, [], covfunc, likfunc,...
    train_features(:,:,2), train_location(:,2));

theta(:,1) = exp([hyp(1).cov(end) hyp(1).cov(1:end-1)']);
theta(:,2) = exp([hyp(2).cov(end) hyp(2).cov(1:end-1)']);
% --- theta(1)     : \sigma_f
% --- theta(2:end) : \sigma_{1,...,p}

sigma2w = exp([hyp(1).lik hyp(2).lik]);

K_1 = CovarianceMatrix_1(train_features(:,:,1),train_features(:,:,1),theta(:,1));
K_1 = K_1+sigma2w(1)*eye(size(K_1,1));

K_2 = CovarianceMatrix_1(train_features(:,:,2),train_features(:,:,2),theta(:,2));
K_2 = K_2+sigma2w(2)*eye(size(K_2,1));

K_1_inv = inv(K_1);
K_2_inv = inv(K_2);

W_1 = diag(theta(2:end,1).^2);
W_2 = diag(theta(2:end,2).^2);

W_1_inv = inv(W_1);
W_2_inv = inv(W_2);

nt_test = size(test_location,1);

%% *                    Apply GPDM to correct the RMSE
progressbar('Correcting RMSE...')
for i=1:nt_test
    test_features_1 = [];
    test_features_2 = [];
    if i<(L+1)
        for j=1:L
            if (i-j)<0
                test_features_1 = [true_test_location(i,1) test_features_1];
                test_features_2 = [true_test_location(i,2) test_features_2];
            else
                test_features_1 = [true_test_location(j,1) test_features_1];
                test_features_2 = [true_test_location(j,2) test_features_2];
            end
        end
    else
        test_features_1 = test_location(i-1:-1:i-L,1)';
        test_features_2 = test_location(i-1:-1:i-L,2)';
    end
    
    % --- \mu_{xstar} and \sigma_{xstar} ---
    %input_mean(:,1) = test_features_1;
    %input_mean(:,2) = test_features_2;
    if i<(L+1)
        input_mean(:,1) = test_features_1;
        input_mean(:,2) = test_features_2;
        input_var(:,:,1) = cov([test_features_1;test_features_1]);
        input_var(:,:,2) = cov([test_features_2;test_features_2]);
    else
        %input_mean(:,1) = true_test_location(i-1:-1:i-L,1)';
        %input_mean(:,2) = true_test_location(i-1:-1:i-L,2)';
        input_mean(:,1) = test_location(i-1:-1:i-L,1)';
        input_mean(:,2) = test_location(i-1:-1:i-L,2)';
        input_var(:,:,1) = cov([test_features_1;...
                                       true_test_location(i-1:-1:i-L,1)']);
        input_var(:,:,2) = cov([test_features_2;...
                                       true_test_location(i-1:-1:i-L,2)']);
    end    
    input_prior_var(1)  = CovarianceMatrix_1(input_mean(:,1)',...
        input_mean(:,1)',theta(:,1));
    input_prior_var(2)  = CovarianceMatrix_1(input_mean(:,2)',...
        input_mean(:,2)',theta(:,2));
    
    % --- calculate vector q ---
    for k=1:nt_train
        feature_diff_1 = input_mean(:,1)-train_features(k,:,1)';
        feature_diff_2 = input_mean(:,2)-train_features(k,:,2)';
        q(k,1) = (det(W_1_inv*input_var(:,:,1)+eye(L)))^(-2)*...
            exp(-0.5*feature_diff_1'*...
            inv(input_var(:,:,1)+W_1)*...
            feature_diff_1);
        
        q(k,2) = (det(W_2_inv*input_var(:,:,2)+eye(L)))^(-2)*...
            exp(-0.5*feature_diff_2'*...
            inv(input_var(:,:,2)+W_2)*...
            feature_diff_2);
    end
    test_location_mean(i,1) = q(:,1)'*K_1_inv*train_location(:,1); %#ok<*MINV>
    test_location_mean(i,2) = q(:,2)'*K_2_inv*train_location(:,2);
    % --- Calculate the matrix Q ---
    for j=1:nt_train
        for k=j:nt_train
            % --- calculate Q_1 ---
            x_temp = (train_features(j,:,1)+train_features(k,:,1))'/2 ...
                -input_mean(:,1);
            x_diff_temp = (train_features(j,:,1)-train_features(k,:,1))';
            Q_temp = det(2*W_1_inv*input_var(:,:,1)+eye(L))^(-2)*...
                exp(-0.5*(x_temp)'*...
                inv(0.5*W_1+input_var(:,:,1))*...
                (x_temp))*...
                exp(-0.5*(x_diff_temp)'*...
                inv(2*W_1)*...
                (x_diff_temp));
            Q_1(j,k) = Q_temp;
            Q_1(k,j) = Q_temp;
            
            % --- calculate Q_2 ---
            x_temp = (train_features(j,:,2)+train_features(k,:,2))'/2 ...
                -input_mean(:,2);
            x_diff_temp = (train_features(j,:,2)-train_features(k,:,2))';
            Q_temp = det(2*W_2_inv*input_var(:,:,2)+eye(L))^(-2)*...
                exp(-0.5*(x_temp)'*...
                inv(0.5*W_2+input_var(:,:,2))*...
                (x_temp))*...
                exp(-0.5*(x_diff_temp)'*...
                inv(2*W_2)*...
                (x_diff_temp));
            Q_2(j,k) = Q_temp;
            Q_2(k,j) = Q_temp;
        end
    end
    
    % --- calculate variance matrix ---
    Ky = K_1_inv*train_location(:,1);
    test_location_var(i,1) = input_prior_var(1) + ...
        trace((Ky*Ky'-inv(K_1))*Q_1) - ...
        trace((q(:,1)'*Ky)*(q(:,1)'*Ky));
    
    Ky = K_2_inv*train_location(:,2);
    test_location_var(i,2) = input_prior_var(2) + ...
        trace((Ky*Ky'-inv(K_2))*Q_2) - ...
        trace((q(:,2)'*Ky)*(q(:,2)'*Ky));
    progressbar(i/nt_test);
end
collapsedTime = toc;
fprintf('Run time: %d minute(s)\n',round(collapsedTime/60));

plot(test_location_mean(:,1),test_location_mean(:,2),'bo--','LineWidth',2);







