# Parameters
struct ModelParams
    β::Float64      # Discount factor
    α::Float64      # Capital share in production
    δ::Float64      # Depreciation rate
    ϕ::Float64      # Adjustment cost parameter
    τ_states::Vector{Float64}   # Tax rate states
    π_τ::Matrix{Float64}        # Transition matrix for τ
    z_states::Vector{Float64}   # Productivity shock states
    π_z::Matrix{Float64}        # Transition matrix for z
    K_grid::Vector{Float64}     # Capital grid
    I_grid::Vector{Float64}     # Lagged investment grid
    num_K::Int
    num_I::Int
    num_τ::Int
    num_z::Int
end


# Function to compute adjustment cost
adjust_cost(I, I_minus, ϕ) = 1 - ϕ * (I / I_minus - 1)^2


# Function to solve the household's problem given pricing functions
function solve_household(params::ModelParams, w::Array{Float64,4}, r::Array{Float64,4}, V_guess::Array{Float64,4}, max_iter, tol)
    # Unpack parameters
    @unpack β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, K_grid, I_grid, num_K, num_I, num_τ, num_z = params
    
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
        
        @threads for τ_z_idx in 1:(num_τ * num_z)
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
                    
                    # Objective function: maximize over L and I
                    function obj(x)
                        L, I = x[1], x[2]
                        if any(x .<= 0)
                            return 1e10
                        end
                        # Compute consumption from budget constraint
                        C = (1 - τ) * w_curr * L + r_curr * K - I
                        if C <= 0
                            return 1e10
                        end

                        # Compute K'
                        K_prime = (1 - δ) * K + adjust_cost(I, I_minus, ϕ) * I

                        # Expected value function
                        π_joint = π_τ[τ_idx, :] * π_z[z_idx, :]'
                        V_future = [V_interp[τp_idx, zp_idx](I, K_prime) for τp_idx in 1:num_τ, zp_idx in 1:num_z]
                        V_expect = sum(π_joint .* V_future)

                        # Current utility
                        utility = log(C) - 0.5 * L^2 + β * V_expect
                        return -utility  # Minimization
                    end
                    
                    # Initial guess for L and I
                    x0 = [L_ss, I_ss]
                    # Bounds for L and I
                    lower_bounds = [1e-6, 1e-6]
                    upper_bounds = [10*L_ss, 10*I_ss]

                    # Optimization
                    result = optimize(obj, lower_bounds, upper_bounds, x0, Fminbox(NelderMead()), Optim.Options(f_tol=tol))
                    L_opt, I_opt = result.minimizer
                    # Compute optimal consumption and K'
                    #C_opt = (1 - τ) * w_curr * L_opt + r_curr * K - I_opt
                    K_prime = (1 - δ) * K + adjust_cost(I_opt, I_minus, ϕ) * I_opt
                    # Update value function and policies
                    V_new[τ_idx, z_idx, I_idx, K_idx] = -result.minimum
                    L_policy[τ_idx, z_idx, I_idx, K_idx] = L_opt
                    I_policy[τ_idx, z_idx, I_idx, K_idx] = I_opt
                    K_policy[τ_idx, z_idx, I_idx, K_idx] = K_prime
                    
            
                    # Update error
                    V_error = max(V_error, abs(V_new[τ_idx, z_idx, I_idx, K_idx] - V[τ_idx, z_idx, I_idx, K_idx]))
                end
            end
        end

        println("Value function error: $V_error")
        # Check convergence
        if V_error < tol
            println("Value function converged.")
            break
        end
        V .= V_new
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
    r_new = α .* Y ./ K_array .- δ                      # Shape: (num_τ, num_z, num_I, num_K)

    return w_new, r_new
end


# Main function to iterate until convergence
function main(V_guess, w_guess, r_guess, K_grid, I_grid, max_iter_V, tol_V, max_iter_price, tol_price)
    # Define parameters and grids
    β = 0.97          # Discount factor
    α = 0.33          # Capital share in production
    δ = 0.1           # Depreciation rate
    ϕ = 0.05          # Adjustment cost parameter

    # Markov chains for τ and z
    τ_states = [0.2, 0.25, 0.3]
    π_τ = [0.9 0.1 0; 0.05 0.9 0.05; 0 0.1 0.9]  # Transition matrix for τ
    z_states = [−0.0673, −0.0336, 0, 0.0336, 0.0673]
    π_z = [0.9727 0.0273 0 0 0;
    0.0041 0.9806 0.0153 0 0;
    0 0.0082 0.9836 0.0082 0;
    0 0 0.0153 0.9806 0.0041;
    0 0 0 0.0273 0.9727] # Transition matrix for z

    num_τ = length(τ_states)
    num_z = length(z_states)
    num_K = length(K_grid)
    num_I = length(I_grid)

    # Initialize parameters struct
    params = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, K_grid, I_grid, num_K, num_I, num_τ, num_z)
    
    # Initial guesses
    w = w_guess
    r = r_guess
    V = V_guess
    L_policy = similar(V)
    I_policy = similar(V)
    K_policy = similar(V)

    for iter = 1:max_iter_price
        println("Main Iteration $iter")
        # Solve household's problem given w and r
        V, L_policy, I_policy, K_policy = solve_household(params, w, r, V_guess, max_iter_V, tol_V)
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
    return V, L_policy, I_policy, K_policy, w , r
end
