folderpath.path = './imagesAcuraRun/merged(K_L_fixedAngle)/';
folderpath.mode = 1;
folderpath.name = 'out_fix';
featureMode.name = 'HISTOGRAM';
featureMode.featureNumber = 64;
output = main_function(folderpath,featureMode);
save output_HIST_64_large_fix