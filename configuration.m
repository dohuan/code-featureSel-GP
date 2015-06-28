function [option,system] = configuration()
%% Setting option or call configuration()
option.x_limit = [-2 2];
option.y_limit = [-2 2];
option.resolution = 0.1;
option.epsilon = 10^(-8);
option.n = 20;      % number of agent
option.m = 2;       % dimension of position
option.q = 1;       % number of measurment per agent
option.colorbar = [-3,3;0,4];
option.counter = 5;
system.sigma2f = 2;
system.sigma2x = 1; % or you can use $system.Sigma2X \in \Real^{nq \times nq}$
system.sigma2v = 0.1;
system.sigma2w = 0.05;

