% Stochastic neoclassical growth model using Chebychev polynomials
% Modified code from Dario Caldara and Jesus Fernandez-Villaverde


%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
%clear all
close all

%----------------------------------------------------------------
% 1. Calibration
%----------------------------------------------------------------

% Technology
alpha = 0.33;                       % Capital Share
beta  = 0.97;                     % Discount factor
delta = 0.1;                    % Depreciation

% Productivity shocks
lambda = 0.95;                     % Persistence parameter of the productivity shock
sigma  = 0.007;                    % S.D. of the productivity shock Z

%----------------------------------------------------------------
% 2. Declare size vectors
%----------------------------------------------------------------

% Productivity and capital
shock_num = 3;                     % number of nodes for technology process Z
node_num = 6;

% Euler errors
grid_num  = 3000;                  % # of grid points for  capital (to compute euler errors)

% Simulation
T         = 10000;                 % Number of periods for the simulation of the economy
dropT     = 1000;                  % Burn-in

%----------------------------------------------------------------
% 3. Steady State + Tauchen 
%----------------------------------------------------------------

% Compute Steady State values

% Initial guesses for k and l
k_guess = 1;
l_guess = 0.3;
x0 = [k_guess; l_guess];

options = optimset('Display','iter','TolFun',1e-12,'TolX',1e-12);
[x_sol, fval, exitflag] = fsolve(@(x) steady_state_system(x,alpha,beta,delta), x0, options);

if exitflag <= 0
    disp('No convergence. Try different initial guesses.');
else
    kss = x_sol(1);
    lss = x_sol(2);
    % Find c_ss
    css = kss^alpha * lss^(1-alpha) - delta * kss;

    % Display results
    fprintf('Steady State Results:\n');
    fprintf('k_ss = %.6f\n', kss);
    fprintf('l_ss = %.6f\n', lss);
    fprintf('c_ss = %.6f\n', css);
end

[Z,PI] = tauchen(shock_num,0,lambda,sigma,3);
%% 

%----------------------------------------------------------------
% 4. Spectral Method using Chebychev Polynomials
%----------------------------------------------------------------

% Define boundaries for capital
cover_grid = 0.25;
k_min = kss*(1-cover_grid);
k_max = kss*(1+cover_grid);
interval = k_max - k_min;

tic

M = node_num*shock_num;

% Find Zeros of the Chebychev Polynomial on order M 
ZC = -cos((2*(1:node_num)'-1)*pi/(2*node_num));

% Define Chebychev polynomials
T_k = ones(node_num,node_num);
T_k(:,2) = ZC;

for i1 = 3:node_num
    T_k(:,i1) = 2*ZC.*T_k(:,i1-1)-T_k(:,i1-2);
end

% Project collocation points in the K space
grid_k = ((ZC+1)*(k_max-k_min))/2+k_min;

% Initial Guess for Chebyshev coefficients
rho_guess = zeros(M,1);
for z_index = 1:shock_num
    rho_guess((z_index-1)*node_num+1)   = lss;
end

% Solve for Chebyshev coefficients
rho = residual_fcn(alpha,beta,delta,k_min,k_max,rho_guess,grid_k,T_k,Z,PI,node_num,shock_num,M);
    
toc

%----------------------------------------------------------------
% 3. Compute Euler Errors and Decision rules
%----------------------------------------------------------------

grid_k_complete = zeros(grid_num,1);

for i = 1:grid_num
    grid_k_complete(i) = k_min+(i-1)*interval/(grid_num-1);
end

[g_k,g_c,g_l,euler_error,max_error]= ...
    eulerr_grid(alpha,beta,delta,rho,Z,PI,...
    k_min,k_max,grid_k_complete,shock_num,node_num,grid_num,M);

[kSeries,cSeries,lSeries,ySeries,eeSeries] = ...
    simulation(alpha,beta,delta,kss,rho,Z,PI,k_min,k_max,node_num,shock_num,M,T,dropT);

mean_error    = sum(eeSeries)/(T-dropT);
max_error_sim = max(eeSeries);

disp(' ')
disp('Integral of Euler Equation Error:')
disp(mean_error)
disp('Max Euler Equation Error Simulation:')
disp(max_error_sim)

%----------------------------------------------------------------
% 4. Figures
%----------------------------------------------------------------
% Define the Matplotlib default colors
colors = [0.1216, 0.4667, 0.7059;   % Light Blue
          1.0000, 0.4980, 0.0549;   % Light Orange
          0.1725, 0.6275, 0.1725];  % Light Green

% Decision Rules
figure(1)
subplot(2,2,1)
hold on;
for i = 1:3
    plot(grid_k_complete, g_c(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(grid_k_complete), max(grid_k_complete)]);
title('Consumption Policy c(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,2)
hold on;
for i = 1:3
    plot(grid_k_complete, g_l(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(grid_k_complete), max(grid_k_complete)]);
title('Labor Policy l(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,3)
hold on;
for i = 1:3
    plot(grid_k_complete, g_k(:,i), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(grid_k_complete), max(grid_k_complete)]);
title("Capital Policy k'(k,z)", 'FontWeight', 'light')
hold off;
saveas(gcf, 'projection_policy.png');

% Euler Equation Error on the Grid
figure(2)
hold on;
for i = 1:3
    plot(grid_k_complete,euler_error(:,i), 'LineWidth', 0.8, 'Color', colors(i,:));
end
xlim([min(grid_k_complete), max(grid_k_complete)]);
hold off;
saveas(gcf, 'projection_eer.png');

% Distribution of simulated variables
[f_c,x_c]   = ksdensity(cSeries);
[f_k,x_k]   = ksdensity(kSeries);
[f_y,x_y]   = ksdensity(ySeries);
[f_l,x_l]   = ksdensity(lSeries);

figure(3)
subplot(2,2,1)
plot(x_c,f_c)
title('Density of Consumption')
subplot(2,2,2)
plot(x_l,f_l)
title('Density of Labor')
subplot(2,2,3)
plot(x_k,f_k)
title('Density of Capital')
subplot(2,2,4)
plot(x_y,f_y)
title('Density of Output')
saveas(gcf, 'projection_dist.png');