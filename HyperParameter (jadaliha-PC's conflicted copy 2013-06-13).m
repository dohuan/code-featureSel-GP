function p = HyperParameter(f,x,p0)
if (nargin < 3)
    theta0 = [1 1 1]'; % sig_f^2, sig_x sig_y
else
    theta0 = p0;
end
itr = 1; t=1;
options = optimset('algorithm','trust-region-reflective','gradobj','on');

p = fminsearch(@(theta) comp_L(x,f,theta),...
    theta0);

% [B.theta(:,t,itr),B.L(t),exitflag] = fmincon(@(theta) comp_L(B.X,B.y,theta),...
%     theta0,[],[],[],[ ],0.001*theta0,10000*theta0,[],options);