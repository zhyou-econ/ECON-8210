function [cFunc, lFunc, kFunc] = compute_policy_functions(oo_, M_)

    % Back out policy functions from perturbation coefficients
    
    % Extract steady states
    y_ss = oo_.dr.ys;
    c_ss = y_ss(strmatch('c', M_.endo_names, 'exact'));
    l_ss = y_ss(strmatch('l', M_.endo_names, 'exact'));
    k_ss = y_ss(strmatch('k', M_.endo_names, 'exact'));
    z_ss = y_ss(strmatch('z', M_.endo_names, 'exact'));
    
    % Extract decision rule coefficients
    g_x    = oo_.dr.ghx;    % 1st order terms
    g_xx   = oo_.dr.ghxx;   % 2nd order terms
    g_xxx  = oo_.dr.ghxxx;  % 3rd order terms
    
    % Identify indices of variables according to oo_.dr.order_var
    c_idx = 3;
    l_idx = 4;
    k_idx = 1;
    z_idx = 2;
    
    % Extract the row of ghx, ghxx, ghxxx for each variable
    g_x_c = g_x(c_idx,:);     g_x_l = g_x(l_idx,:);     g_x_k = g_x(k_idx,:);
    g_xx_c = g_xx(c_idx,:);   g_xx_l = g_xx(l_idx,:);   g_xx_k = g_xx(k_idx,:);
    g_xxx_c = g_xxx(c_idx,:); g_xxx_l = g_xxx(l_idx,:); g_xxx_k = g_xxx(k_idx,:);
    
    % Now construct anonymous functions:
    % Let dx = k - k_ss; dz = z - z_ss;
    % 1st order: g_x * x
    % 2nd order: (1/2)*g_xx * [dx^2; dx*dz; dz^2]
    % 3rd order: (1/6)*g_xxx * [dx^3; dx^2*dz; dx*dz^2; dz^3]
    
    cFunc = @(K,Z) ...
        c_ss + ...
        (g_x_c(1)*(K-k_ss) + g_x_c(2)*(Z-z_ss)) + ...
        0.5*(g_xx_c(1)*(K-k_ss)^2 + g_xx_c(2)*(K-k_ss)*(Z-z_ss) + g_xx_c(3)*(Z-z_ss)^2) + ...
        (1/6)*(g_xxx_c(1)*(K-k_ss)^3 + g_xxx_c(2)*(K-k_ss)^2*(Z-z_ss) + g_xxx_c(3)*(K-k_ss)*(Z-z_ss)^2 + g_xxx_c(4)*(Z-z_ss)^3);

    lFunc = @(K,Z) ...
        l_ss + ...
        (g_x_l(1)*(K-k_ss) + g_x_l(2)*(Z-z_ss)) + ...
        0.5*(g_xx_l(1)*(K-k_ss)^2 + g_xx_l(2)*(K-k_ss)*(Z-z_ss) + g_xx_l(3)*(Z-z_ss)^2) + ...
        (1/6)*(g_xxx_l(1)*(K-k_ss)^3 + g_xxx_l(2)*(K-k_ss)^2*(Z-z_ss) + g_xxx_l(3)*(K-k_ss)*(Z-z_ss)^2 + g_xxx_l(4)*(Z-z_ss)^3);

    kFunc = @(K,Z) ...
        k_ss + ...
        (g_x_k(1)*(K-k_ss) + g_x_k(2)*(Z-z_ss)) + ...
        0.5*(g_xx_k(1)*(K-k_ss)^2 + g_xx_k(2)*(K-k_ss)*(Z-z_ss) + g_xx_k(3)*(Z-z_ss)^2) + ...
        (1/6)*(g_xxx_k(1)*(K-k_ss)^3 + g_xxx_k(2)*(K-k_ss)^2*(Z-z_ss) + g_xxx_k(3)*(K-k_ss)*(Z-z_ss)^2 + g_xxx_k(4)*(Z-z_ss)^3);
end
