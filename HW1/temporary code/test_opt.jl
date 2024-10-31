using NLsolve, LinearAlgebra, Statistics, Optim, Interpolations, Parameters, Plots, Base.Threads, JLD2, Distributions, Random
include("q6_ge_func.jl")
#include("q6_ge_func_stochastic.jl")
include("IRF_func.jl")

# Parameterization
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


# Q6.2
# Solve the steady state
function solve_ss!(f, ss)
    c, l, k, i, g = ss

    f[1] = 1 - (1 - δ) * β - α * k^(α - 1) * l^(1 - α) * β
    f[2] = l - 0.75 * (1 - α) * k^α * l^(-α) * 1/c
    f[3] = c + i + g - k^α * l^(1 - α)
    f[4] = g - 0.25 * (1 - α) * k^α * l^(1 - α)
    f[5] = i - δ * k
end

# Initial guess for steady state
ss0 = 0.5 * ones(5)

# Solve for steady state using NLsolve
solution = nlsolve(solve_ss!, ss0)

# Extract the solution
C_ss, L_ss, K_ss, I_ss, G_ss = solution.zero
w_ss = K_ss^α * L_ss^(-α) * (1 - α)
r_ss = K_ss^(α - 1) * L_ss^(1 - α) * α

num_K = 20
num_I = 10
K_grid = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K))
I_grid = collect(range(0.5 * I_ss, 1.5 * I_ss, length=num_I))
# Initialize parameters struct
params = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)

# Intialize guess for V, w, r
#=
V_guess = ones(num_τ, num_z, num_I, num_K) .* (log(C_ss) - 0.5 * L_ss^2)
=#


V_guess = ones(num_τ, num_z, num_I, num_K)
for τ_idx in 1:num_τ
    τ = τ_states[τ_idx]
    for z_idx in 1:num_z
        z = z_states[z_idx]
        for I_prev_idx in 1:num_I
            for K_idx in 1:num_K
                K = K_grid[K_idx]
                V_guess[τ_idx, z_idx, I_prev_idx, K_idx] = (log(C_ss) - 0.5 * L_ss^2) + 0.1 * log(1 + K_idx + I_prev_idx)
            end
        end
    end
end


w = ones(num_τ, num_z, num_I, num_K) .* w_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)
r = ones(num_τ, num_z, num_I, num_K) .* r_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)


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

# Function to compute adjustment cost
adjust_cost(I, I_minus, ϕ) = 1 - ϕ * (I / I_minus - 1)^2


# Define the objective function for the static labor choice
function compute_L_opt(τ, w_curr, r_curr, K, I)
    a = (1 - τ) * w_curr
    b = r_curr * K - I
    discriminant = b^2 + 4 * a^2
    L_opt = (-b + sqrt(discriminant)) / (2 * a)

    return L_opt
end


max_iter = 100
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
                
                #=
                # Objective function: maximize over L and I
                function obj_I(I_vec)
                    I = I_vec[1]

                    if I < 0.5*I_ss || I > 1.5*I_ss
                        return Inf
                    end

                    L_opt = compute_L_opt(τ, w_curr, r_curr, K, I)

                    # Compute consumption
                    C = (1 - τ) * w_curr * L_opt + r_curr * K - I
                    if C  <= 0
                        return Inf
                    end

                    # Compute K'
                    K_prime = (1 - δ) * K + adjust_cost(I, I_minus, ϕ) * I

                    if K_prime  < 0.7 * K_ss || K_prime  > 1.3 * K_ss
                        return Inf
                    end

                    # Compute expected utility
                    EV = 0.0
                    for τp_idx in 1:num_τ
                        p_τ = π_τ[τ_idx, τp_idx]
                        for zp_idx in 1:num_z
                            p_z = π_z[z_idx, zp_idx]
                            # Interpolate V_old at (I, Kp)
                            V_future = V_interp[τp_idx, zp_idx](I, K_prime)
                            EV += p_τ * p_z * V_future
                        end
                    end

                    # Discounted utility (new value function)
                    utility = (1 - β) * (log(C) - 0.5 * L_opt^2) + β * EV
                    return -utility  # Minimization
                end
                

                # Optimization with initial guess I_ss
                # result = optimize(obj_I, [0.5 * I_ss], [1.5 * I_ss], [I_ss], Fminbox(NelderMead()), Optim.Options(f_tol=tol))
                result = optimize(obj_I, [I_minus])
                I_opt = result.minimizer[1]
                =#



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

                function find_opt_I(I_grid)
                    max_value = -Inf
                    I_opt = I_ss
                    L_opt = L_ss  # Default value
                    K_prime = K_ss

                    for I_candidate in I_grid
                        
                        L = compute_L_opt(τ, w_curr, r_curr, K, I_candidate)
                        
                        # Compute consumption from budget constraint
                        C = (1 - τ) * w_curr * L + r_curr * K - I_candidate
                        if C <= 0
                            continue  # Infeasible consumption, skip to next I_candidate
                        end

                        # Capital accumulation
                        K_prime = (1 - δ) * K + adjust_cost(I_candidate, I_minus, ϕ) * I_candidate
            
                        EV = compute_expected_value(I_candidate, K_prime)
                        
                        total_value = (1 - β) * (log(C) - 0.5 * L^2) + β * EV
                        
                        # Update the maximum value and corresponding I and L
                        if total_value > max_value
                            max_value = total_value
                            I_opt = I_candidate
                            L_opt = L
                        end
                    end 

                    return max_value, I_opt, L_opt, K_prime
                end

                # Compute L_opt given I_opt
                # L_opt = compute_L_opt(τ, w_curr, r_curr, K, I_opt)
                # Compute optimal consumption and K'
                # K_prime = (1 - δ) * K + adjust_cost(I_opt, I_minus, ϕ) * I_opt
                # Update value function and policies
                # V_new[τ_idx, z_idx, I_idx, K_idx] = -result.minimum

                max_value, I_opt, L_opt, K_prime = find_opt_I(I_grid)
                V_new[τ_idx, z_idx, I_idx, K_idx] = max_value
                L_policy[τ_idx, z_idx, I_idx, K_idx] = L_opt
                I_policy[τ_idx, z_idx, I_idx, K_idx] = I_opt
                K_policy[τ_idx, z_idx, I_idx, K_idx] = K_prime
               
                # Update error
                V_error = max(V_error, abs(V_new[τ_idx, z_idx, I_idx, K_idx] - V[τ_idx, z_idx, I_idx, K_idx]))
            end
        end
    end
    
    # Update value function
    V .= V_new

    println("Value function error: $V_error")
    # Check convergence
    if V_error < 10^-3
        println("Value function converged.")
        break
    end
            
end


#=
τ_idx=1
z_idx=5
I_idx=num_I
I_minus=I_grid[I_idx]
K_idx=num_K
K=K_grid[K_idx]
τ = τ_states[τ_idx]


w_curr = w[τ_idx, z_idx, I_idx, K_idx]
r_curr = r[τ_idx, z_idx, I_idx, K_idx]
 
# Create interpolations for the current value function
for τ_idx = 1:num_τ
    for z_idx = 1:num_z
        V_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid), V[τ_idx, z_idx, :, :], extrapolation_bc=Line())
    end
end

I_candidate = I_grid[10]

L = compute_L_opt(τ, w_curr, r_curr, K, I_candidate)

# Compute consumption from budget constraint
C = (1 - τ) * w_curr * L + r_curr * K - I_candidate

# Capital accumulation
K_prime = (1 - δ) * K + adjust_cost(I_candidate, I_minus, ϕ) * I_candidate


function compute_expected_value(I_candidate, K_prime)
    EV = 0.0
    for τp_idx in 1:num_τ
        p_τ = π_τ[τ_idx, τp_idx]
        for zp_idx in 1:num_z
            p_z = π_z[z_idx, zp_idx]
            # Interpolate V_old at (I_candidate, K_prime)
            V_future = V_interp[τp_idx, zp_idx](I_candidate, K_prime)
            EV += p_τ * p_z * V_future
        end
    end
    return EV
end


#EV1 = compute_expected_value(I_candidate, K_prime)
#total_value1 = (1 - β) * (log(C) - 0.5 * L^2) + β * EV1


        
function find_opt_I(I_grid)
    max_value = -Inf
    I_opt = I_ss
    L_opt = L_ss  # Default value
    K_prime = K_ss
    
    for I_candidate in I_grid
        
        L = compute_L_opt(τ, w_curr, r_curr, K, I_candidate)
        
        # Compute consumption from budget constraint
        C = (1 - τ) * w_curr * L + r_curr * K - I_candidate
        if C <= 0
            continue  # Infeasible consumption, skip to next I_candidate
        end

        # Capital accumulation
        K_prime = (1 - δ) * K + adjust_cost(I_candidate, I_minus, ϕ) * I_candidate

        EV = compute_expected_value(I_candidate, K_prime)
        
        total_value = (1 - β) * (log(C) - 0.5 * L^2) + β * EV
        
        # Update the maximum value and corresponding I and L
        if total_value > max_value
            max_value = total_value
            I_opt = I_candidate
            L_opt = L
        end

    end
    
    return max_value, I_opt, L_opt, K_prime
end

max_value, I_opt, L_opt, K_prime = find_opt_I(I_grid)
=#