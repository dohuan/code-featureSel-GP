function handels = plotpredictedmaps( map , option )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

nf = size(map,3);
A = squeeze(num2cell(map,(1:2)));
h_f = figure;
handels = imdisp(A,'Border',[0.1 0.15],'map','jet');
bound = [min(min(min(map))) max(max(max(map)))];
tmp2 = (option.x_limit(1):option.resolution:option.x_limit(2));
tmp1 = (option.y_limit(1):option.resolution:option.y_limit(2));
for index1 = 1:nf
    h_o = handels(index1);
    h_a = get(h_o,'parent');
    
    set(h_o,'XData',tmp2,'YData',tmp1);   % add X and Y scale
    
    set(h_a,...
        'Visible','on',...
        'YDir','normal',...
        'CLim',bound,...
        'XLim',[tmp2(1),tmp2(end)],...
        'Ylim',[tmp1(1),tmp1(end)]);
    
    set(h_a,...
        'Position', get(h_a,'Position').*[0.90,1,0.90,1]);
end
switch nf
    case 8
        h = axes('CLim',bound,'Visible','off','Position',[0,0.05,1.05,0.90]);
    case 16
        h = axes('CLim',bound,'Visible','off','Position',[0,0.05,1.00,0.90]);
    otherwise
        h = axes('CLim',bound,'Visible','off','Position',[0,0.05,1.05,0.90]);

end
colorbar('peer',h);

% h_f = get(h_a,'parent');
window_size = get(h_f,'Position')+[0 0 200 0];
set(h_f,'Position',window_size,'Name','MAP');



end

