# Parameters
mutable struct ModelParams
    β::Float64      # Discount factor
    α::Float64      # Capital share in production
    δ::Float64      # Depreciation rate
    ϕ::Float64      # Adjustment cost parameter
    τ_states::Vector{Float64}   # Tax rate states
    π_τ::Matrix{Float64}        # Transition matrix for τ
    z_states::Vector{Float64}   # Productivity shock states
    π_z::Matrix{Float64}        # Transition matrix for z
    μ_τ::Float64 # AR1 parameters for tax shock
    ρ_τ::Float64 # AR1 parameters for tax shock
    σ_τ::Float64 # AR1 parameters for tax shock
    μ_z::Float64 # AR1 parameters for TFP shock
    ρ_z::Float64 # AR1 parameters for TFP shock
    σ_z::Float64 # AR1 parameters for TFP shock
    K_ss::Float64               # Steady state capital level
    K_grid::Vector{Float64}     # Capital grid
    I_grid::Vector{Float64}     # Lagged investment grid
    num_K::Int
    num_I::Int
    num_τ::Int
    num_z::Int
end


# Function to compute adjustment cost
adjust_cost(I, I_minus, ϕ) = 1 - ϕ * (I / I_minus - 1)^2


# Function to solve the static labor choice as a function of I
function compute_L_opt(τ, w, r, K, I, initial_guess)
    # Define the equation as a function of L
    eq(L) = (1 - τ) * w / ((1 - τ) * w * L + r * K - I) + 0.2 / L - L
    
    # Use find_zero from Roots.jl to solve for L
    L_solution = find_zero(eq, initial_guess)
    return L_solution
end


# Function to solve the household's problem given pricing functions
function solve_household(params::ModelParams, w::Array{Float64,4}, r::Array{Float64,4}, V_guess::Array{Float64,4}, max_iter, tol, mode)
    # Unpack parameters
    @unpack β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z = params
    
    # Initialize value function and policy functions
    V = V_guess
    V_new = similar(V)
    L_policy = similar(V)
    I_policy = similar(V)
    K_policy = similar(V)
    
    # Create interpolations for V
    V_interp = Array{Any}(undef, num_τ, num_z)
    
    
    for iter = 1:max_iter
        println("Value Function Iteration $iter")
        V_error = 0.0

        # Create interpolations for the current value function
        for τ_idx = 1:num_τ
            for z_idx = 1:num_z
                V_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid), V[τ_idx, z_idx, :, :], extrapolation_bc=Line())
            end
        end   

        for τ_z_idx in 1:(num_τ * num_z)
            τ_idx = div(τ_z_idx - 1, num_z) + 1
            z_idx = mod(τ_z_idx - 1, num_z) + 1
            τ = τ_states[τ_idx]

            for I_idx = 1:num_I
                I_minus = I_grid[I_idx]
                for K_idx = 1:num_K
                    K = K_grid[K_idx]
                    # Get current wage and rental rate
                    w_curr = w[τ_idx, z_idx, I_idx, K_idx]
                    r_curr = r[τ_idx, z_idx, I_idx, K_idx]
                    
                    # Function to compute EV
                    function compute_expected_value(I_eval, K_p)
                        EV = 0.0
                        for τp_idx in 1:num_τ
                            p_τ = π_τ[τ_idx, τp_idx]
                            for zp_idx in 1:num_z
                                p_z = π_z[z_idx, zp_idx]
                                # Interpolate V_old at (I_candidate, K_prime)
                                V_future = V_interp[τp_idx, zp_idx](I_eval, K_p)
                                EV += p_τ * p_z * V_future
                            end
                        end
                        return EV
                    end

                    # Objective function: maximize over L and I
                    function obj_I(I_vec)
                        I = I_vec[1]
                        if I <= minimum(I_grid) || I >= maximum(I_grid)
                            return 1e10
                        end
                        L_opt = compute_L_opt(τ, w_curr, r_curr, K, I, L_ss)
                        # Compute consumption
                        C = (1 - τ) * w_curr * L_opt + r_curr * K - I
                        if C <= 0
                            return 1e10  # Penalize infeasible consumption
                        end
                        # Compute government expenditure
                        G =  τ * w_curr * L_opt
                        # Compute K'
                        K_prime = (1 - δ) * K + adjust_cost(I, I_minus, ϕ) * I
                        # Expected value function
                        V_expect = compute_expected_value(I, K_prime)
                        # Discounted utility (new value function)
                        utility = (1 - β) * (log(C) + 0.2 * log(G) - 0.5 * L_opt^2) + β * V_expect
                        return -utility  # Minimization
                    end
                    
                    if mode == 1 #Max every period
                        # Optimization with initial guess I_ss
                        # result = optimize(obj_I, [0.5 * I_ss], [1.5 * I_ss], [I_ss], Fminbox(NelderMead()), Optim.Options(f_tol=tol))
                        result = optimize(obj_I, [I_ss], Optim.Options(f_tol=tol))
                        I_opt = result.minimizer[1]
                        # Compute L_opt given I_opt
                        L_opt = compute_L_opt(τ, w_curr, r_curr, K, I_opt, L_ss)
                        # Compute optimal consumption and K'
                        K_prime = (1 - δ) * K + adjust_cost(I_opt, I_minus, ϕ) * I_opt
                        # Update value function and policies
                        V_new[τ_idx, z_idx, I_idx, K_idx] = -result.minimum
                        L_policy[τ_idx, z_idx, I_idx, K_idx] = L_opt
                        I_policy[τ_idx, z_idx, I_idx, K_idx] = I_opt
                        K_policy[τ_idx, z_idx, I_idx, K_idx] = K_prime
                    elseif mode == 2
                        if mod(iter, 10) == 1
                            # Optimization with initial guess I_ss
                            result = optimize(obj_I, [I_ss], Optim.Options(f_tol=tol))
                            I_opt = result.minimizer[1]
                            # Compute L_opt given I_opt
                            L_opt = compute_L_opt(τ, w_curr, r_curr, K, I_opt, L_ss)
                            # Compute optimal consumption and K'
                            K_prime = (1 - δ) * K + adjust_cost(I_opt, I_minus, ϕ) * I_opt
                            # Update value function and policies
                            V_new[τ_idx, z_idx, I_idx, K_idx] = -result.minimum
                            L_policy[τ_idx, z_idx, I_idx, K_idx] = L_opt
                            I_policy[τ_idx, z_idx, I_idx, K_idx] = I_opt
                            K_policy[τ_idx, z_idx, I_idx, K_idx] = K_prime
                        else
                            V_new[τ_idx, z_idx, I_idx, K_idx] = -obj_I(I_policy[τ_idx, z_idx, I_idx, K_idx])
                        end
                    end
                    # Update error
                    V_error = max(V_error, abs(V_new[τ_idx, z_idx, I_idx, K_idx] - V[τ_idx, z_idx, I_idx, K_idx]))
                end
            end
        end
        
        # Update value function
        V .= V_new

        println("Value function error: $V_error")
        # Check convergence
        if V_error < tol
            println("Value function converged.")
            break
        end
        
    end
    
    return V, L_policy, I_policy, K_policy
end


function update_prices(params::ModelParams, L_policy::Array{Float64,4})
    # Unpack parameters
    @unpack α, δ, τ_states, z_states, K_grid, num_τ, num_z, num_I, num_K = params

    # Reshape arrays for broadcasting
    z_array = reshape(z_states, (1, num_z, 1, 1))  # Shape: (1, num_z, 1, 1)
    K_array = reshape(K_grid, (1, 1, 1, num_K))    # Shape: (1, 1, 1, num_K)
    
    # Broadcast operations over L_policy
    L = L_policy  # Alias for clarity, has shape: (num_τ, num_z, num_I, num_K)
    
    # Compute production output Y using broadcasting
    Y = exp.(z_array) .* K_array .^ α .* L .^ (1 - α)  # Shape: (num_τ, num_z, num_I, num_K)
    
    # Compute w_new and r_new using broadcasting
    w_new = (1 - α) .* Y ./ L                           # Shape: (num_τ, num_z, num_I, num_K)
    r_new = α .* Y ./ K_array                           # Shape: (num_τ, num_z, num_I, num_K)

    return w_new, r_new
end


# Main function to iterate until convergence
function main(params::ModelParams, V_guess, w_guess, r_guess, max_iter_V, tol_V, max_iter_price, tol_price, mode)
    
    # Initial guesses
    w = w_guess
    r = r_guess
    V = V_guess
    L_policy = similar(V)
    I_policy = similar(V)
    K_policy = similar(V)
    K_grid_gen = params.K_grid

    for iter = 1:max_iter_price
        println("Main Iteration $iter")
        # Solve household's problem given w and r
        V, L_policy, I_policy, K_policy = solve_household(params, w, r, V_guess, max_iter_V, tol_V, mode)
        # Update pricing functions
        w_new, r_new = update_prices(params, L_policy)
        # Compute errors
        w_error = maximum(abs.(w_new .- w))
        r_error = maximum(abs.(r_new .- r))
        println("Pricing function errors: w_error = $w_error, r_error = $r_error")
        # Check convergence
        if max(w_error, r_error) < tol_price
            println("Pricing functions converged.")
            break
        end
        # Update pricing functions
        w .= w_new
        r .= r_new
    end
    
    println("Computation completed.")
    # The policy functions are stored in L_policy, I_policy, K_policy
    return V, L_policy, I_policy, K_policy, w , r, K_grid_gen
end
