folderpath.path = './imagesAcuraRun/merged(K_L)/';
folderpath.mode = 1;
folderpath.name = 'out_unfix';
featureMode.name = 'HISTOGRAM';
featureMode.featureNumber = 64;
output = main_function(folderpath,featureMode);
save output_HIST_64_large