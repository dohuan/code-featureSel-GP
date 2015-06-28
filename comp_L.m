% function [L,pL] = comp_L(X,y,theta)
function L = comp_L(X,y,theta)
theta = [theta(1), theta(2), theta(3), inf, 0.01]'; % sig_f^2, sig_x, sig_y, sig_t, sig_w
n = length(y);


sig_w = theta(5);

K = comp_K(X,X,theta);
C = K + sig_w^2 * eye(n);
% Ci = inv(C);
% Cd = det(C);

v = C\y;
L = -1/2*y'*v - 1/2*logdet(C) - n/2*log(2*pi);
L = -L;

% 
% pC = comp_pC(X,X,K,theta);
% pL = zeros(m,1);
% for j = 1:m
%     tr = 0;
%     for i = 1:n
%         tr = tr + Ci(i,:)*pC(:,i,j);
%     end
%     pL(j) = 0.5*v'*pC(:,:,j)*v - 0.5*tr;
% end
% pL = -[pL(1,1);pL(3,1)];
