function l_val = lOfkz(k, z, jz, theta, PsiFun)
% Computes the labor supply level given basis functions and their coefficients.

    psi_vals = PsiFun(k);
    l_val = psi_vals*theta(:,jz);
end
