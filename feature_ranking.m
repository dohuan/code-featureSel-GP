function [forward_rank, backward_rank] = ...
    feature_ranking( Predicted_Layer_Map,...
                     Prediction_Layer_Variance,...
                     features, location, option)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

n = size(features,1);
nf = size(features,2);
ng = numel(Predicted_Layer_Map)/nf;
map = reshape(Predicted_Layer_Map,ng,nf);
map_var = reshape(Prediction_Layer_Variance,ng,nf);
forward_rank = zeros(1,nf);
backward_rank= zeros(1,nf);
tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
ng1 = size(tmp1,2);

features_diffrence = (kron(ones(n,1),map) - kron(features,ones(ng,1))).^2;
features_variance  = ones(n*ng,nf) * diag(option.hyperparametrs(4,:)) + ...
                       kron(ones(n,1),map_var);
features_likelihood= log(features_variance) + ...
                        features_diffrence./features_variance;
cost = sum(features_likelihood,2);
% Compute RMS Error using all predicted map
total_cost = reshape(cost,ng,n);
[~,I0] = min(total_cost);
I0 = I0-1;
IX = floor(I0/ng1)+1;    
IY = mod(I0,ng1)+1;
p0 = [tmp2(IX);tmp1(IY)]';
total_low_rms = sqrt(sum(sum((p0 - location(:,1:2)).^2))/n);
    
for index2 = 1 : nf   
    % Compute RMS Error using just index2-th predicted map layer
    forward_cost = reshape(features_likelihood(:,index2),ng,n);
    [~,I1] = min(forward_cost);
    I1 = I1-1;
    IX = floor(I1/ng1)+1;    
    IY = mod(I1,ng1)+1;
    p1 = [tmp2(IX);tmp1(IY)]';
    forward_rms = sqrt(sum(sum((p1 - location(:,1:2)).^2))/n);
    forward_rank(index2) = forward_rms;
    
    % Compute RMS Error using all except just index2-th predicted map layer
    backward_cost= reshape(cost,ng,n) - forward_cost;
    [~,I2] = min(backward_cost);
    I2 = I2-1;
    IX = floor(I2/ng1)+1;
    IY = mod(I2,ng1)+1;
    p2 = [tmp2(IX);tmp1(IY)]';
    backward_rms = sqrt(sum(sum((p2 - location(:,1:2)).^2))/n);   
    backward_rank(index2) = backward_rms;
     
end

% Compute RMS Error without using predicted map
p0 = ones(n,1) * [median(tmp2),median(tmp1)];
total_high_rms = sqrt(sum(sum((p0 - location(:,1:2)).^2))/n);
     

   % plotscore(forward_rank,backward_rank,total_low_rms,total_high_rms)
end

function plotscore(forward_rank,backward_rank,total_low_rms,total_high_rms)
    nf = numel(forward_rank);
    figure1 = figure('Name','Feature Ranking Score','NumberTitle','off');
    % Create axes
    axes1 = axes('Parent',figure1);
    box(axes1,'on');
    grid(axes1,'on');
    hold(axes1,'all');

    % Create multiple lines using matrix input to plot
    plot1 = plot((1:nf),forward_rank,'r',...
                 (1:nf),backward_rank,'b',...
                 'Parent',axes1,'MarkerEdgeColor',[0 0 0],...
        'LineWidth',2);
    set(plot1(1),'MarkerFaceColor',[0 0.498039215803146 0],'Marker','square',...
        'LineStyle',':',...
        'Color',[1 0 0],...
        'DisplayName','Forward Score');
    set(plot1(2),'MarkerFaceColor',[1 1 0],'MarkerSize',8,'Marker','o',...
        'LineStyle',':',...
        'Color',[0 0 1],...
        'DisplayName','Backward Score');

    % Create xlabel
    xlabel('Feature Index');

    % Create ylabel
    ylabel('RMS Error');

    % Create legend
    legend1 = legend(axes1,'show');
    set(legend1,...
        'Position',[0.773809523809524 0.671738959610248 0.0910714285714286 0.0462046204620462]);

    plot([1,nf],[total_high_rms,total_high_rms],'r',...
         [1,nf],[total_low_rms,total_low_rms],'b');
end