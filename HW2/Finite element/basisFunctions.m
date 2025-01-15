function psi_vals = basisFunctions(k, k_grid)
    % basisFunctions evaluates the piecewise linear basis functions at point k.
    %
    % The basis functions are defined as follows:
    %   For i=1: support is [k_1, k_2]
    %       Psi_1(k) linearly falls from 1 at k_1 to 0 at k_2, and is 0 elsewhere.
    %
    %   For i=2,...,8: support is [k_{i-1}, k_{i+1}]
    %       Psi_i(k) = (k - k_{i-1})/(k_i - k_{i-1}) for k in [k_{i-1}, k_i]
    %                 = (k_{i+1}-k)/(k_{i+1}-k_i)   for k in [k_i,k_{i+1}]
    %       and 0 elsewhere.
    %
    %   For i=9: support is [k_8, k_9]
    %       Psi_9(k) linearly rises from 0 at k_8 to 1 at k_9, and is 0 elsewhere.
    %
    % Inputs:
    %   k:      Scalar at which to evaluate the basis functions.
    %   k_grid: Vector of knots [k_1, k_2, ..., k_n]. Must have at least 10 knots if using i=9 as stated.
    %
    % Output:
    %   psi_vals: 1-by-Nbasis vector of basis function values at point k.

    Nk = length(k_grid);
    % If we have 9 basis functions, then Nbasis = 9, hence we need Nk=10 knots.
    Nbasis = Nk;
    psi_vals = zeros(1, Nbasis);

    % i=1: support [k_1, k_2]
    i = 1;
    if k >= k_grid(1) && k <= k_grid(2)
        % Psi_1: 0 at k_1, 1 at k_2
        psi_vals(i) = (k_grid(2) - k) / (k_grid(2) - k_grid(1));
    else
        psi_vals(i) = 0;
    end

    % i=2,...,8: support [k_{i-1}, k_{i+1}]
    % For i in [2,...,8]:
    %   On [k_{i-1}, k_i]: rises from 0 to 1
    %   On [k_i, k_{i+1}]: falls from 1 to 0
    for i = 2:Nk-1
        left = k_grid(i-1);
        mid  = k_grid(i);
        right= k_grid(i+1);

        if k >= left && k <= mid
            psi_vals(i) = (k - left)/(mid - left);
        elseif k >= mid && k <= right
            psi_vals(i) = (right - k)/(right - mid);
        else
            psi_vals(i) = 0;
        end
    end

    % i=9: support [k_8, k_9]
    i = Nk;
    if k >= k_grid(Nk-1) && k <= k_grid(Nk)
        % Psi_9: 1 at k_8, 0 at k_9
        psi_vals(i) = (k - k_grid(Nk-1))/(k_grid(Nk) - k_grid(Nk-1));
    else
        psi_vals(i) = 0;
    end
end
