# Initial conditions
C_sim = zeros(T)
L_sim = zeros(T)
I_sim = zeros(T)
w_sim = zeros(T)
r_sim = zeros(T)
K_sim = zeros(T+1)
τ_sim = fill(τ_ss, T)
z_sim = fill(0.0, T)
I_prev_sim = zeros(T+1)

z_sim[1] += 0.036
K_sim[1] = K_ss
I_prev_sim[1] = δ * K_ss

for t in 1:T
    τ_t = τ_sim[t]
    z_t = z_sim[t]
    
    # Find the closest τ and z in the grids
    τ_t_closest_idx = argmin(abs.(τ_states .- τ_t))
    z_t_closest_idx = argmin(abs.(z_states .- z_t))
    τ_t_closest = τ_states[τ_t_closest_idx]
    z_t_closest = z_states[z_t_closest_idx]
    
    # Get current states
    I_prev = I_prev_sim[t]
    K = K_sim[t]
    
    # Find indices for I_prev and K
    I_prev_idx = searchsortedfirst(I_grid, I_prev)
    K_idx = searchsortedfirst(K_grid, K)
    I_prev_idx = clamp(I_prev_idx, 1, num_I)
    K_idx = clamp(K_idx, 1, num_K)
    
    # Retrieve policy variables
    L_t = L_policy[τ_t_closest_idx, z_t_closest_idx, I_prev_idx, K_idx]
    I_t = I_policy[τ_t_closest_idx, z_t_closest_idx, I_prev_idx, K_idx]
    C_t = C_policy[τ_t_closest_idx, z_t_closest_idx, I_prev_idx, K_idx]
    w_t = w_func[τ_t_closest_idx, z_t_closest_idx, I_prev_idx, K_idx]
    r_t = r_func[τ_t_closest_idx, z_t_closest_idx, I_prev_idx, K_idx]
    
    # Store variables
    C_sim[t] = C_t
    L_sim[t] = L_t
    I_sim[t] = I_t
    w_sim[t] = w_t
    r_sim[t] = r_t
    
    # Update states for next period
    adjustment_cost = ϕ * ((I_t / I_prev - 1)^2)
    K_sim[t+1] = (1 - δ) * K + (1 - adjustment_cost) * I_t
    I_prev_sim[t+1] = I_t
    
end

# Compute deviations from steady state
C_irf = (C_sim .- C_ss) ./ C_ss
L_irf = (L_sim .- L_ss) ./ L_ss
I_irf = (I_sim .- I_ss) ./ I_ss
K_irf = (K_sim .- K_ss) ./ K_ss
w_irf = (w_sim .- w_ss) ./ w_ss
r_irf = (r_sim .- r_ss) ./ r_ss

time = 1:T

# Create a figure with 3 rows and 2 columns
plot_layout = @layout [a b; c d; e f]
p1 = plot(layout = plot_layout, size=(800, 600))

# Plot each variable's IRF in a subplot
plot!(p1[1], time, C_irf, label=nothing, title="Consumption", xlabel="Time", ylabel="Deviation")
plot!(p1[2], time, L_irf, label=nothing, title="Labor", xlabel="Time", ylabel="Deviation")
plot!(p1[3], time, I_irf, label=nothing, title="Investment", xlabel="Time", ylabel="Deviation")
plot!(p1[4], time, w_irf, label=nothing, title="Wage", xlabel="Time", ylabel="Deviation")
plot!(p1[5], time, r_irf, label=nothing, title="Interest Rate", xlabel="Time", ylabel="Deviation")
plot!(p1[6], time, K_irf[1:T], label=nothing, title="Capital", xlabel="Time", ylabel="Deviation")