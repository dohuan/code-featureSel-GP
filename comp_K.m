function K = comp_K(X1,X2,theta)

% data size
n1 = size(X1,1);
n2 = size(X2,1);

% hyperparameters
sig_f = theta(1);
sig_x = theta(2);
sig_y = theta(3);
sig_t = theta(4);

% calculate K
K = zeros(n1,n2);
for i = 1:n1
    for j = 1:n2
        K(i,j) = sig_f^2 * exp(-0.5*(X1(i,1)-X2(j,1))^2/sig_x^2)...
            *exp(-0.5*(X1(i,2)-X2(j,2))^2/sig_y^2)...
            *exp(-0.5*(X1(i,3)-X2(j,3))^2/sig_t^2);
    end
end























