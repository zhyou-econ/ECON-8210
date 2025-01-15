% Plot policy functions by third-order perturbation

% Parametrization
lambda = 0.95;
sigma  = 0.007;
% Markov productivity states
[z_nodes, Pi] = tauchen(3,0,lambda,sigma,3);
z_vals = z_nodes; % {z_l,z_m,z_h}

% Compute policy functions
[cFunc, lFunc, kFunc] = compute_policy_functions(oo_, M_);

% K space
k_ss = oo_.dr.ys(3,1);
k_range = linspace(0.75*k_ss,1.25*k_ss,3000);

% Evaluate policy functions at z_l, z, z_h
c_to_plot = zeros(3,size(k_range,2));
l_to_plot = zeros(3,size(k_range,2));
kp_to_plot = zeros(3,size(k_range,2));
for i = 1:size(k_range,2)
    k = k_range(i);
    for j = 1:3
        z = z_nodes(j);
        c_to_plot(j,i) = cFunc(k, z);
        l_to_plot(j,i) = lFunc(k, z);
        kp_to_plot(j,i) = kFunc(k, z);
    end
end

% Define the Matplotlib default colors
colors = [0.1216, 0.4667, 0.7059;   % Light Blue
          1.0000, 0.4980, 0.0549;   % Light Orange
          0.1725, 0.6275, 0.1725];  % Light Green

figure(1)
subplot(2,2,1)
hold on;
for i = 1:3
    plot(k_range, c_to_plot(i,:), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_range), max(k_range)]);
title('Consumption Policy c(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,2)
hold on;
for i = 1:3
    plot(k_range, l_to_plot(i,:), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_range), max(k_range)]);
title('Labor Policy l(k,z)', 'FontWeight', 'light')
hold off;

subplot(2,2,3)
hold on;
for i = 1:3
    plot(k_range, kp_to_plot(i,:), 'LineWidth', 1, 'Color', colors(i,:));
end
xlim([min(k_range), max(k_range)]);
title("Capital Policy k'(k,z)", 'FontWeight', 'light')
hold off;

saveas(gcf, 'perturbation_policy.png');