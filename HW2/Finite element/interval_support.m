function [kl, kr] = interval_support(i, k_grid)
    % Interval support depending on i:
    % i=1: [k_1, k_2]
    % i=2,...,8: [k_{i-1}, k_{i+1}]
    % i=9: [k_8, k_9]

    Nbasis = size(k_grid,1);

    if i == 1
        kl = k_grid(1);
        kr = k_grid(2);
    elseif i == Nbasis
        kl = k_grid(Nbasis-1);
        kr = k_grid(Nbasis);
    else
        kl = k_grid(i-1);
        kr = k_grid(i+1);
    end
end