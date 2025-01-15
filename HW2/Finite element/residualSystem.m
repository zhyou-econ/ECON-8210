function Rval = residualSystem(th, k_grid, z_vals, Pi, beta, alpha, delta, PsiFun, xi, wgt, Nbasis, Nz)
    % th is a vector of length Nbasis*Nz with coefficients \theta_i^j
    theta = reshape(th, Nbasis, Nz);
    Rval = zeros(Nbasis * Nz, 1);

    % For each state of z and each basis function i, we form the integral:
    %   ∫ Psi_i(k)*R(k,z^j;theta) dk = 0
    %
    % Where the integration domain depends on i:
    % i=1: integrate over [k_1, k_2]
    % i=2,...,8: integrate over [k_{i-1}, k_{i+1}]
    % i=9: integrate over [k_8, k_9]

    for jz = 1:Nz
        for i = 1:Nbasis
            [kl, kr] = interval_support(i, k_grid);

            % Gauss-Legendre integration on [kl, kr]
            kmid = (kl + kr)/2;
            half_len = (kr - kl)/2;
            integral_val = 0;

            for iq = 1:length(xi)
                kq = kmid + half_len * xi(iq);
                psi_vals = PsiFun(kq);
                psi_i = psi_vals(i);

                R_ = computeResidual(kq, z_vals(jz), jz, k_grid, z_vals, Pi, beta, alpha, delta, theta, PsiFun);

                integral_val = integral_val + psi_i * R_ * wgt(iq);
            end

            integral_val = integral_val * half_len;
            eq_index = (jz - 1)*Nbasis + i;
            Rval(eq_index) = Rval(eq_index) + integral_val;
        end
    end
end