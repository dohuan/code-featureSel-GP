function str = custom_sprintf( p1,p2,p3,p4 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

str = 'Estimated position = (            )  vs True position = (            )';
s1 = sprintf('%2.1f',p1); str(27-size(s1,2)+1:27) = s1;
s2 = sprintf('%2.1f',p2); str(33-size(s2,2)+1:33) = s2;
s3 = sprintf('%2.1f',p3); str(62-size(s3,2)+1:62) = s3;
s4 = sprintf('%2.1f',p4); str(68-size(s4,2)+1:68) = s4;
disp(str)
end

