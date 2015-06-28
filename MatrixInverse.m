function Ainv=MatrixInverse(A)
% A=rand(7,7); 
%
% Ainv1=inv(A);
% Ainv2=MatrixInverse(A);
%
% Ainv1-Ainv2
%

% Do Lu decompositon to obtain triangle matrices (can easily be inverted)
[L,U,P] = Lu(A);

% Solve linear system for Identity matrix
I=eye(size(A));
s=size(A,1);
Ainv=zeros(size(A));
for i=1:s
    b=I(:,i);
    Ainv(:,i)=TriangleBackwardSub(U,TriangleForwardSub(L,P*b));
end


function [L,U,P] = Lu(A)
%  LU factorization.
%
%   
% [L,U,P] = Lu(A) returns unit lower triangular matrix L, upper
% triangular matrix U, and permutation matrix P so that P*A = L*U.
%
% example,
%   A=rand(9,9);
%
%   [L,U,P] = Lu(A);
%   sum(sum(abs(P*A- L*U)))
%   
%   [L,U,P] = lu(A);
%   sum(sum(abs(P*A- L*U)))
%
%
s=length(A);
U=A;
L=zeros(s,s);
PV=(0:s-1)';
for j=1:s,
    % Pivot Voting (Max value in this column first)
    [~,ind]=max(abs(U(j:s,j)));
    ind=ind+(j-1);
    t=PV(j); PV(j)=PV(ind); PV(ind)=t;
    t=L(j,1:j-1); L(j,1:j-1)=L(ind,1:j-1); L(ind,1:j-1)=t;
    t=U(j,j:end); U(j,j:end)=U(ind,j:end); U(ind,j:end)=t;

    % LU
    L(j,j)=1;
    for i=(1+j):size(U,1)
       c= U(i,j)/U(j,j);
       U(i,j:s)=U(i,j:s)-U(j,j:s)*c;
       L(i,j)=c;
    end
end
P=zeros(s,s);
P(PV(:)*s+(1:s)')=1;

function C=TriangleBackwardSub(U,b)
% Triangle Matrix Backward Substitution
%
% Solves C = U \ B;
%
%          |1|         |2 2 1|
% With b = |2| and U = |0 1 4|
%          |3|         |0 0 3|
%
s=length(b);
C=zeros(s,1);
C(s)=b(s)/U(s,s);
for j=(s-1):-1:1
    C(j)=(b(j) -sum(U(j,j+1:end)'.*C(j+1:end)))/U(j,j);
end

function C=TriangleForwardSub(L,b)
% Triangle Matrix Forward Substitution
%
% Solves C = L \ b
%
%          |1|         |1 0 0|
% With b = |2| and L = |2 1 0|
%          |3|         |3 4 1|
%
s=length(b);
C=zeros(s,1);
C(1)=b(1)/L(1,1);
for j=2:s
    C(j)=(b(j) -sum(L(j,1:j-1)'.*C(1:j-1)))/L(j,j);
end

