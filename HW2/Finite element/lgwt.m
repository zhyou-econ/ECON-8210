function [x, w] = lgwt(N, a, b)
% lgwt(N, A, B) computes the Legendre-Gauss nodes and weights on the interval [A,B].
% It approximates the integral:
%   \int_a^b f(x) dx ≈ sum_{i=1}^N w_i f(x_i),
% where x_i are the Legendre-Gauss nodes and w_i are their corresponding weights.
%
% Inputs:
%   N : Number of nodes (integer > 0)
%   a : Lower bound of the interval
%   b : Upper bound of the interval
%
% Outputs:
%   x : N x 1 column vector of nodes
%   w : N x 1 column vector of weights
%
% Reference:
% This code is adapted from the implementation by Greg von Winckel (2004).

% Check that N is positive
if N < 1
    error('N must be at least 1.');
end

% The initial guess of roots is based on Chebyshev-Gauss-Lobatto nodes:
N = N-1;
N1 = N+1; N2 = N+2;
xu = linspace(-1,1,N1)';
y = cos((2*(0:N)'+1)*pi/(2*N+2)) + (0.27/N1)*sin(pi*xu*N/N2);

% Allocate memory for Legendre-Gauss Vandermonde matrix and derivative
L = zeros(N1, N2);
Lp = zeros(N1, N2);

% Newton-Raphson iteration for improved accuracy
y0 = 2; % start with a difference larger than machine precision
while max(abs(y - y0)) > eps
    y0 = y;

    % Compute Legendre-Gauss Vandermonde matrix
    L(:,1) = 1;
    Lp(:,1) = 0;

    L(:,2) = y;
    Lp(:,2) = 1;

    for k = 2:N+1
        L(:,k+1) = ((2*k-1)*y.*L(:,k) - (k-1)*L(:,k-1))/k;
    end

    % Lp: derivative w.r.t. y of L(:,N2)
    Lp = N2*(L(:,N1)-y.*L(:,N2))./(1 - y.^2);

    % Newton step
    y = y0 - L(:,N2)./Lp;
end

% Transform nodes from [-1,1] to [a,b]
x = (a*(1 - y) + b*(1 + y))/2;

% Compute weights
w = (b - a)./((1 - y.^2).*Lp.^2) * (N2/N1)^2;

% If any NaNs occur, it may be due to numerical instability.
% Check for NaNs and correct if necessary:
if any(isnan(w))
    warning('NaN encountered in weights. Attempting to resolve numerically...');
    idx = isnan(w);
    w(idx) = 0; % This is a fallback measure; ideally no NaNs should occur.
end

if any(isnan(x))
    warning('NaN encountered in nodes. Attempting to resolve numerically...');
    idx = isnan(x);
    x(idx) = 0; % Ideally, this should not happen.
end

end
