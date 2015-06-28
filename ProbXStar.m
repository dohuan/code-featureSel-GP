function [zhatxstar sigma2xstar] = ProbXStar(xstar,X,Y)
[option,system] = configuration();
if isfield(option,'time')
    d=3;
else
    d=2;
end
X = reshape(X',[],1);    
X = reshape(X,d,[])';

sigma2w = system.sigma2w;
n = size(X,1);
KxstarX = CovarianceMatrix(xstar,X);
KXX = CovarianceMatrix(X);
Kxstarxstar = CovarianceMatrix(xstar);
Lambda = (KXX+sigma2w*eye(n))^(-1);
zhatxstar = KxstarX*Lambda*Y;
sigma2xstar = Kxstarxstar - KxstarX * Lambda * KxstarX';