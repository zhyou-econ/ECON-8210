function F = steady_state_system(x,alpha,beta,delta)

% Solves for steady state. 

    k = x(1);
    l = x(2);

    % Equation 1 (Euler condition):
    eq1 = alpha * k^(alpha-1) * l^(1 - alpha) + 1 - delta - (1/beta);

    % Equation 2 (Labor-consumption condition combined with resource constraint):
    lhs = (1 - alpha)*k^alpha / l^(alpha+1);
    rhs = k^alpha * l^(1-alpha) - delta * k;
    eq2 = lhs - rhs;

    F = [eq1; eq2];
end