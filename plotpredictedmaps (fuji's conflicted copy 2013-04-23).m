function plotpredictedmaps( map , option )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
map_size = size(map);
tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));

numberofaxes = 2^ceil(log(map_size(3))/log(2));
switch numberofaxes
    case{4}
        
    case{8}
        
    case{16}  
        figNaive = figure(  'Name','Predicted map',...
                            'Position',[100 50 400 670],...
                            'NumberTitle','off');    
        for index1 = 1: map_size(3) 
            number_of_col = 8;
            IX = floor((index1 - 1)/number_of_col)+1;
            IY = mod  ((index1 - 1),number_of_col)+1;
            handels.axis(index1) = ...
                axes('Parent',figNaive,...
                 'Position',[0.11 0.55 0.85 0.37],...
                 'DataAspectRatio',[1 1 1]);
            handels.plot(index1) = ...
                image(tmp2,tmp1,map(:,:,1),...
                 'DisplayName','Naive Conditional Mean',...
                 'CDataMapping','scaled');
            
        end
    otherwise
end

                 
axes('Parent',figNaive,'Position',[0.11 0.55 0.85 0.37]);


imagesc(tmp1 ,tmp2,map(:,:,1),'DisplayName','Naive Conditional Mean')
title('Predicted field');
colorbar

       
imagesc(tmp1 ,tmp2,sigma2xstar,'DisplayName','Naive Conditional Variance')        
hold on
tmp = X;
plot(tmp(:,1)',tmp(:,2)','Color',[1 0 1],'MarkerSize',10,'Marker','x'...
    ,'LineWidth',3,'LineStyle','none','DisplayName','Real Position');    
title('Prediction variance');
colorbar

end

