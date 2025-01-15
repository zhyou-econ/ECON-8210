clear; close all; clc;

%% Parameters
beta    = 0.97;
alpha   = 0.33;
delta   = 0.1;
lambda  = 0.95;
sigma   = 0.007;

% Number of z states (3-state approximation)
Nz = 3;

% We have 9 knots for capital to form 8 intervals.
Nk = 9; 
Nbasis = Nk;

%% Declare size vectors
% Productivity and capital
shock_num = 3;                     % number of nodes for technology process Z

% Euler errors
grid_num  = 3000;                  % # of grid points for  capital (to compute euler errors)

% Simulation
T         = 10000;                 % Number of periods for the simulation of the economy
dropT     = 1000;                  % Burn-in

%% Steady state (placeholder)
% Initial guesses for k and l
k_guess = 1;
l_guess = 0.3;
x0 = [k_guess; l_guess];

options = optimset('Display','iter','TolFun',1e-12,'TolX',1e-12);
[x_sol, fval, exitflag] = fsolve(@(x) steady_state_system(x,alpha,beta,delta), x0, options);

if exitflag <= 0
    disp('No convergence. Try different initial guesses.');
else
    k_ss = x_sol(1);
    l_ss = x_sol(2);
    % Find c_ss
    c_ss = k_ss^alpha * l_ss^(1-alpha) - delta * k_ss;

    % Display results
    fprintf('Steady State Results:\n');
    fprintf('k_ss = %.6f\n', k_ss);
    fprintf('l_ss = %.6f\n', l_ss);
    fprintf('c_ss = %.6f\n', c_ss);
end

k_min = 0.75 * k_ss;
k_max = 1.25 * k_ss;
% Equal grid
k_grid = linspace(k_min,k_max,Nk)'; % k_1,...,k_9

interval = k_max - k_min;

%% Tauchen's discretization of z
[z_nodes, Pi] = tauchen(shock_num,0,lambda,sigma,3);
z_vals = z_nodes; % {z_l,z_m,z_h}

%% Basis Functions
Psi = @(k) basisFunctions(k, k_grid);

%% Gaussian Legendre
% Compute nodes and weights for [-1, 1]
[xi, wgt] = lgwt(10, -1, 1);

%% Initial Guess
% theta is a Nbasis-by-Nz matrix of coefficients
theta0 = repmat(l_ss, Nbasis*Nz, 1);

%% Solve for theta0
options = optimset('Display','Iter','TolFun',10^(-15),'TolX',10^(-15));
theta_sol = fsolve(@(theta) residualSystem(theta, k_grid, z_vals, Pi, beta, alpha, delta, Psi, xi, wgt, Nbasis, Nz),theta0,options);

% Display solution
disp('Solution for Theta:');
disp(reshape(theta_sol, Nbasis, Nz));

%% Compute policy function and euler residuals
% theta_sol: A vector of length Nbasis*Nz containing the coefficients.
theta = reshape(theta_sol, Nbasis, Nz);

% Preallocate arrays for policies and residuals
L_policy = zeros(length(k_grid), Nz);     % Labor policy l(k,z)
Kp_policy = zeros(length(k_grid), Nz);    % Next-period capital policy k'(k,z)
C_policy = zeros(length(k_grid), Nz);     % Consumption c(k,z)
R_residuals = zeros(length(k_grid), Nz);  % Euler residuals R(k,z)

% Compute policies and residuals on the capital grid for each z-state
for jz = 1:Nz
    z_current = z_vals(jz);
    for ik = 1:length(k_grid)
        k_val = k_grid(ik);
        
        % Compute labor l(k,z)
        l_val = lOfkz(k_val, z_current, jz, theta, Psi);
        
        % Ensure positivity if needed
        if l_val <= 0
            l_val = eps;
        end
        
        % Compute consumption c(k,z)
        c_val = (1 - alpha)*exp(z_current)*k_val^alpha * l_val^(-alpha-1);
        if c_val <= 0
            c_val = eps;
        end
        
        % Compute next period capital k'(k,z)
        kp_val = exp(z_current)*k_val^(alpha)*l_val^(1 - alpha) + (1 - delta)*k_val - c_val;
        if kp_val <= 0
            kp_val = eps;
        end
        
        % Compute Euler residual R(k,z;theta)
        R_val = eer(k_val, z_current, jz, k_grid, z_vals, Pi, beta, alpha, delta, theta, Psi);

        % Store results
        L_policy(ik, jz) = l_val;
        C_policy(ik, jz) = c_val;
        Kp_policy(ik, jz) = kp_val;
        R_residuals(ik, jz) = R_val;
    end
end


%% Plotting Decision Rule
% Define the Matplotlib default colors
colors = [0.1216, 0.4667, 0.7059;   % Light Blue
          1.0000, 0.4980, 0.0549;   % Light Orange
          0.1725, 0.6275, 0.1725];  % Light Green

figure(1)
subplot(2,2,1)
hold on;
for i = 1:3
    plot(k_grid, C_policy(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_grid), max(k_grid)]);
title('Consumption Policy c(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,2)
hold on;
for i = 1:3
    plot(k_grid, L_policy(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_grid), max(k_grid)]);
title('Labor Policy l(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,3)
hold on;
for i = 1:3
    plot(k_grid, Kp_policy(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_grid), max(k_grid)]);
title("Capital Policy k'(k,z)", 'FontWeight', 'light')
hold off;

saveas(gcf, 'FE_policy.png');

%% Plotting Euler Residuals
figure(2)
hold on;
for i = 1:3
    plot(k_grid, R_residuals(:,i), 'LineWidth', 1.5, 'Color', colors(i,:));
end
xlim([min(k_grid), max(k_grid)]);
hold off;
saveas(gcf, 'FE_eer.png');