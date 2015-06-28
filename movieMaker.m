clear
clc
close all
%%

features(1).name = 'ABS_FFT';
features(1).data_type{1} = 'out_fix';
features(1).nf = 64; % for one axis, actual nf = nf*2

features(2).name = 'HISTOGRAM';
features(2).data_type{1} = 'out_fix';
features(2).nf = 64;

features(3).name = 'WAVELET';
features(3).data_type{1} = 'out_fix';
features(3).nf = 72;

outputCount = 1;
for i=1:size(features,2)
    for j=1:size(features(i).data_type,2)
        features_temp.name = features(i).name;
        features_temp.nf = features(i).nf;
        features_temp.data_type = features(i).data_type{j};
        output_temp = main_function(features_temp);
        output(outputCount) = output_temp;
        outputCount = outputCount + 1;
    end
end

n_case = size(output,2);

save('./result/movie_run_062815');

