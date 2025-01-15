function ee = eer(k, z, jz, k_grid, z_vals, Pi, beta, alpha, delta, theta, PsiFun)
    % Compute the log Euler error

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

    R = 1- c_current*beta*Ezterm;
    ee = log10(abs(R));
end