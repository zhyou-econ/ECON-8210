function [x, w] = gaussHermiteQuadrature(N)
% gaussHermiteQuadrature computes the nodes (x) and weights (w) for the
% N-point Gauss-Hermite quadrature using the Golub-Welsch algorithm.

    i = (1:N-1)';
    beta = sqrt(i/2);
    alpha = zeros(N,1);

    % Construct the symmetric tridiagonal matrix J:
    J = diag(alpha) + diag(beta,1) + diag(beta,-1);

    % Compute eigen decomposition
    [V,D] = eig(J);
    [x,idx] = sort(diag(D));   % nodes are eigenvalues
    V = V(:,idx);

    % The weights are given by:
    % w_i = c * (v_1i)^2, where v_1i is the first element of the i-th eigenvector.
    % For Gauss-Hermite, the factor c = sqrt(pi).
    w = (V(1,:).^2)' * sqrt(pi);



end
