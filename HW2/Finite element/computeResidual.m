function R_ = computeResidual(k, z, jz, k_grid, z_vals, Pi, beta, alpha, delta, theta, PsiFun)
    % Compute the residual:
    % R(k,z;theta) = c(k,z)^(-1) - beta * sum_z' pi(z'|z) c(k',z')^{-1} [1 - delta + alpha e^{z'} k'^{alpha-1} l(k',z')^{1-alpha}]
    %
    % Here, l(k,z) = sum_i theta_i^z Psi_i(k)
    % Ensure l>0, c>0, k'>0 to avoid undefined values.

    l_current = lOfkz(k, z, jz, theta, PsiFun);
    l_current = max(l_current, eps);
    c_current = (1 - alpha)*exp(z)*k^alpha*l_current^(-alpha-1);
    c_current = max(c_current, eps);
    k_prime = exp(z)*k^alpha*l_current^(1 - alpha) + (1 - delta)*k - c_current;
    k_prime = max(k_prime, eps);

    Ezterm = 0;
    for jp = 1:length(z_vals)
        z_next = z_vals(jp);
        l_next = lOfkz(k_prime, z_next, jp, theta, PsiFun);
        l_next = max(l_next, eps);
        c_next = (1 - alpha)*exp(z_next)*k_prime^alpha*l_next^(-alpha-1);
        c_next = max(c_next, eps);

        marg_term = c_next^(-1)*(1 - delta + alpha*exp(z_next)*k_prime^(alpha-1)*l_next^(1 - alpha));
        Ezterm = Ezterm + Pi(jz,jp)*marg_term;
    end

    R_ = c_current^(-1) - beta*Ezterm;
end