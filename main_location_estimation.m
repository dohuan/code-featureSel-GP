clc
clear
option.x_limit = [0 4];
option.y_limit = [0 4];
option.resolution = 0.1;

load('8Layer-Map.mat');%load trained map

testpoint_features = onlineimage('testimage.jpg');

cost_function = Predicted_8Layer_Map;
for i=1:8
    cost_function(:,:,i) = cost_function(:,:,i) - testpoint_features(i);
end

summedup_cost_function = sum(cost_function.^2,3);

[C1 I1] = min(summedup_cost_function);
[C2 I2] = min(C1);

tmp1 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp2 = (option.y_limit(1):option.resolution:option.y_limit(2));
p1 = tmp1(I1(I2));
p2 = tmp2(I2);
disp(sprintf('Estimated position: = ( %2.1f \t %2.1f)',p1,p2));
