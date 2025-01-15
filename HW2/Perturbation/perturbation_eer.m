% EER for perturbation method

% Initialization
k_grid = k_range;
euler_residual = nan(3, length(k_grid));
N = 100;
[xi, wi] = gaussHermiteQuadrature(N); %use Gaussian Hermite Quadrature to compute the integral

% Compute Euler equation residuals
for j = 1:3
    z_current = z_nodes(j);
    for i=1:length(k_grid)
        k_current = k_grid(i);
        c_current = c_to_plot(j,i);
        l_current = l_to_plot(j,i);
        k_next = kp_to_plot(j,i);
        
        % LHS
        LHS = 1/c_current;
        % RHS
        RHS = 0;
        for k = 1:N
            z_next = lambda*z_current + sigma*sqrt(2)*xi(k);
            c_next = cFunc(k_next,z_next);
            l_next = lFunc(k_next,z_next);
            RHS = RHS + wi(k)* beta*((alpha*exp(z_next)*k_next^(alpha-1)*l_next^(1-alpha)+ (1 - delta))*(1/c_next));
        end
        RHS = RHS/sqrt(pi);

        euler_residual(j, i) = log10(abs(1 - RHS/LHS));
    end
end

% Plot Euler equation residuals
% Define the Matplotlib default colors
colors = [0.1216, 0.4667, 0.7059;   % Light Blue
          1.0000, 0.4980, 0.0549;   % Light Orange
          0.1725, 0.6275, 0.1725];  % Light Green

figure;
hold on;
for i = 1:3
    plot(k_grid, euler_residual(i,:), 'LineWidth', 1.5, 'Color', colors(i,:));
end
xlim([min(k_grid), max(k_grid)]);
hold off;
saveas(gcf, 'perturbation_eer.png');