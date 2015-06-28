clear
clc
close all
paper_name = 'ACC'; % ACC, IMAVIS
%% ===================== Setups for ACC paper =============================

if strcmp(paper_name,'ACC')
    tic
    %addpath(genpath('./matlabPyrTools/')) % Pyr toolbox
    %addpath(genpath('./gpml-matlab/'))    % GPML toolbox
    
    features(1).name = 'ABS_FFT';
    features(1).data_type{1} = 'in';
    features(1).data_type{2} = 'out_unfix';
    features(1).data_type{3} = 'out_fix';
    features(1).nf = 64; % for one axis, actual nf = nf*2
    
    features(2).name = 'HISTOGRAM';
    features(2).data_type{1} = 'in';
    features(2).data_type{2} = 'out_unfix';
    features(2).data_type{3} = 'out_fix';
    features(2).nf = 64;
    
    features(3).name = 'WAVELET';
    features(3).data_type{1} = 'in';
    features(3).data_type{2} = 'out_unfix';
    features(3).data_type{3} = 'out_fix';
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
    
    fid = fopen(['./result/output_' paper_name '.csv'],'w');
    fprintf(fid,'Case,rmse_all_t,rmse_all_v,Opt_f,rmse_opt_t,rmse_opt_v,BIC_All,BIC_opt');
    fprintf(fid,'\n');
    for i=1:n_case
        fprintf(fid,              [output(i).name ',' ...
            num2str(output(i).rmse_all) ',' ...
            num2str(output(i).rmse_all_v) ',' ...
            num2str(output(i).opt_features) ',' ...
            num2str(output(i).rmse_selected_test) ',' ...
            num2str(output(i).rmse_selected_valid) ',' ...
            num2str(output(i).BIC_all) ',' ...
            num2str(output(i).BIC_selected)]);
        fprintf(fid,'\n');
    end
    fclose(fid);
    timeCollapsed = toc/60;
    fprintf('Collapsed time: %f minutes',timeCollapsed);
end
%% ==================== Setups for IMAVIS journal =========================

if strcmp(paper_name,'IMAVIS')
    tic
    addpath(genpath('./matlabPyrTools/')) % Pyr toolbox
    addpath(genpath('./gpml-matlab/'))    % GPML toolbox
    % ----------------- folder path ------------------------
    folderpath(1).path = './images2/';
    folderpath(2).path = './imagesAcuraRun/merged(K_L)/';
    folderpath(3).path = './imagesAcuraRun/merged(K_L_fixedAngle)/';
    
    folderpath(1).mode = 0;
    folderpath(2).mode = 1;
    folderpath(3).mode = 1;
    
    folderpath(1).name = 'in';
    folderpath(2).name = 'out_unfix';
    folderpath(3).name = 'out_fix';
    
    % ----------------- feature mode ------------------------
    featureMode(1).name = 'ABS_FFT';
    featureMode(2).name = 'HISTOGRAM';
    featureMode(3).name = 'WAVELET';
    
    featureMode(1).featureNumber = 64;
    featureMode(2).featureNumber = 64;
    featureMode(3).featureNumber = 72;
    
    n_folder = size(folderpath,2);
    n_feature = size(featureMode,2);
    outputCount = 1;
    for i=1:n_feature  % i: index for featureMode
        for j=1:n_folder  % j: index for file folders
            f_count_temp = size(featureMode(i).featureNumber,2);
            for k=1:f_count_temp
                fModeTemp.name = featureMode(i).name;
                fModeTemp.featureNumber = featureMode(i).featureNumber(k);
                output_temp = main_f(folderpath(j),fModeTemp);
                output_temp.name = ...
                    [fModeTemp.name '-' ...
                    num2str(fModeTemp.featureNumber) '-' ...
                    folderpath(j).name];
                output(outputCount) = output_temp;
                
                outputCount = outputCount + 1;
            end
        end
    end
    
    n_case = size(output,2);
    save ./main_f/output.mat
    fid = fopen(['./main_f/output_' paper_name '.csv'],'w');
    fprintf(fid,'Case,rmse_all,Opt_f,rmse_opt_t,rmse_opt_v,BIC_All,BIC_opt');
    fprintf(fid,'\n');
    for i=1:n_case
        fprintf(fid,              [output(i).name ',' ...
            num2str(output(i).rmse_all) ',' ...
            num2str(output(i).opt_features) ',' ...
            num2str(output(i).rmse_selected_test) ',' ...
            num2str(output(i).rmse_selected_valid) ',' ...
            num2str(output(i).BIC_all) ',' ...
            num2str(output(i).BIC_selected)]);
        fprintf(fid,'\n');
    end
    fclose(fid);
    timeCollapsed = toc/60;
    fprintf('Collapsed time: %f minutes',timeCollapsed);
end
