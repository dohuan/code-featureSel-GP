function [ output_args ] = dropbox()
%dropbox:   This function returns the main directory of dropbox on
%           different computer. You may add computername and dropbox
%           path here.


pairs = {... %% put computer name and dropbox path here!
    {'FUJI',            'h:\dropbox'},...
    {'DECS-PC',         'C:\Users\jadaliha\Dropbox'},...
    };


computername = getenv('COMPUTERNAME');
for index = 1:size(pairs,2)
    if strcmp(computername,pairs{1,index}{1,1})
        output_args = pairs{1,index}{1,2};
    end
end

end

