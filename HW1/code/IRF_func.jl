# Compute consumption policy function based on budget constraint
function compute_consumption_policy(params::ModelParams, L_policy, I_policy, w_func, r_func)
    # Unpack parameters
    @unpack τ_states, K_grid, num_K, num_I, num_τ, num_z = params
    
    # Initialize the consumption policy function
    C_policy = zeros(num_τ, num_z, num_I, num_K)

    # Loop over each state in the grid
    for τ_idx in 1:num_τ
        τ = τ_states[τ_idx]
        for z_idx in 1:num_z
            for I_idx in 1:num_I
                for K_idx in 1:num_K
                    # Retrieve the policy values
                    L = L_policy[τ_idx, z_idx, I_idx, K_idx]
                    I = I_policy[τ_idx, z_idx, I_idx, K_idx]
                    w = w_func[τ_idx, z_idx, I_idx, K_idx]
                    r = r_func[τ_idx, z_idx, I_idx, K_idx]
                    K = K_grid[K_idx]

                    # Apply the budget constraint to compute C
                    C_policy[τ_idx, z_idx, I_idx, K_idx] = (1 - τ) * w * L + r * K - I
                end
            end
        end
    end

    return C_policy
end


# Back out the underlying AR1 process
function compute_ar1_coefficients(P::Matrix{Float64}, s::Vector{Float64})

    N = size(P, 1)

    # Compute the stationary distribution π
    eigvals, eigvecs = eigen(P')
    stationary_index = argmin(abs.(eigvals .- 1)) # Find index of eigenvalue closest to 1
    π_stat = abs.(eigvecs[:, stationary_index])
    π_stat/= sum(π_stat)  # Normalize to sum to 1

    #Calculate the mean μ and variance σ^2 of the Markov chain states under the stationary distribution
    μ = sum(π_stat .* s)
    σ = sqrt(sum(π_stat .* (s .- μ) .^ 2))

    #Calculate the covariance Cov(X_t, X_{t+1})
    covariance = sum(π_stat[i] * P[i, j] * (s[i] - μ) * (s[j] - μ) for i in 1:N, j in 1:N)

    #Estimate the AR(1) persistence parameter ρ
    ρ = covariance / σ^2

    return μ, ρ, σ
end


# simulate the impulse response
function simulate_irf(params::ModelParams, shock_type::Symbol, T::Int, C_policy_interp, L_policy_interp, I_policy_interp, w_func_interp, r_func_interp)
    # Unpack parameters
    @unpack K_ss, ϕ, δ, τ_states, z_states, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z  = params

    # Initialize arrays to store variables
    C_sim = zeros(T)
    L_sim = zeros(T)
    I_sim = zeros(T)
    w_sim = zeros(T)
    r_sim = zeros(T)
    K_sim = zeros(T+1)
    τ_sim = fill(μ_τ, T)
    z_sim = fill(μ_z, T)
    I_prev_sim = zeros(T+1)
    
    # Apply the shock at t = 1
    if shock_type == :tax
        τ_sim = [μ_τ + σ_τ * ρ_τ^(t-1) for t in 1:T]
    elseif shock_type == :technology
        z_sim = [μ_z + σ_z * ρ_z^(t-1) for t in 1:T]
    else
        error("Invalid shock type. Use :tax or :technology.")
    end
    
    # Initial conditions
    K_sim[1] = K_ss
    I_prev_sim[1] = δ * K_ss

    for t in 1:T
        τ_t = τ_sim[t]
        z_t = z_sim[t]
        
        # Get current states
        I_prev = I_prev_sim[t]
        K = K_sim[t]
               
        # Retrieve policy variables
        L_t = L_policy_interp[τ_t, z_t, I_prev, K]
        I_t = I_policy_interp[τ_t, z_t, I_prev, K]
        C_t = C_policy_interp[τ_t, z_t, I_prev, K]
        w_t = w_func_interp[τ_t, z_t, I_prev, K]
        r_t = r_func_interp[τ_t, z_t, I_prev, K]
        
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
    
    return C_irf, L_irf, I_irf, K_irf[1:T], w_irf, r_irf
end