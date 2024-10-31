using NLsolve, LinearAlgebra, Statistics, Optim, Interpolations, Parameters, Plots, Base.Threads, JLD2, Distributions, Random, Roots
include("q6_ge_func.jl")
include("IRF_func.jl")

# Parameterization
β = 0.97          # Discount factor
α = 0.33          # Capital share in production
δ = 0.1           # Depreciation rate
ϕ = 0.05          # Adjustment cost parameter

# Markov chains for τ and z
τ_states = [0.2, 0.25, 0.3]
τ_ss = 0.25
π_τ = [0.9 0.1 0; 0.05 0.9 0.05; 0 0.1 0.9]  # Transition matrix for τ
z_states = [−0.0673, −0.0336, 0, 0.0336, 0.0673]
π_z = [0.9727 0.0273 0 0 0;
0.0041 0.9806 0.0153 0 0;
0 0.0082 0.9836 0.0082 0;
0 0 0.0153 0.9806 0.0041;
0 0 0 0.0273 0.9727] # Transition matrix for z
num_τ = length(τ_states)
num_z = length(z_states)

μ_τ, ρ_τ, σ_τ = compute_ar1_coefficients(π_τ, τ_states)
μ_z, ρ_z, σ_z = compute_ar1_coefficients(π_z, z_states)


"""
6.2 Solving the steady state
"""

#Normalize K_ss=1
K_ss = 1
# Define the function to find the root of
function steady_state(L)
    # Production function
    Y = L^(1 - α)
    # Wage rate
    w = (1 - α)* L^(-α)
    # Consumption
    C = Y * (1 - (1 - α) * τ_ss) - δ
    # Left-hand side of the labor FOC
    LHS = L
    # Right-hand side of the labor FOC
    RHS = (1 - τ_ss) * w / C
    # Return the difference
    return LHS - RHS
end

# Use a root-finding algorithm to solve for L
result = nlsolve(x -> steady_state(x[1]), [0.8])  # Initial guess L=0.8
L_ss = result.zero[1]

# Compute steady-state values
Y_ss = K_ss^α * L_ss^(1 - α)
w_ss = (1 - α) * K_ss^α * L_ss^(-α)
r_ss = α * K_ss^(α - 1) * L_ss^(1 - α)
I_ss = δ * K_ss
G_ss = τ_ss * w_ss * L_ss
C_ss = Y_ss - I_ss - G_ss
println([K_ss, L_ss, Y_ss, C_ss, I_ss, G_ss, w_ss, r_ss])



"""
6.3  VFI with fixed grid
"""

num_K = 250
num_I = 50
K_grid = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K))
I_grid = collect(range(0.5 * I_ss, 1.5 * I_ss, length=num_I))
# Initialize parameters struct
params = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)

# Intialize guess for V, w, r
V_guess = ones(num_τ, num_z, num_I, num_K) .* (log(C_ss) + 0.2 * log(G_ss) - 0.5 * L_ss^2)
w_guess = ones(num_τ, num_z, num_I, num_K) .* w_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)
r_guess = ones(num_τ, num_z, num_I, num_K) .* r_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)

# Set max iterations and tolerance of price
max_iter_price = 10
tol_price = 1e-3
time_taken = @elapsed begin
    # First find the Eq price with max_iter_V = 20 and tol_V = 1e-3
    V_temp, L_policy_temp, I_policy_temp, K_policy_temp, w_temp , r_temp = main(params, V_guess, w_guess, r_guess, 20, 1e-3, max_iter_price, tol_price, 1)
    # Then find the associated value function with max_iter_V = 1000 and tol_V = 1e-6
    V, L_policy, I_policy, K_policy, w_func , r_func = main(params, V_temp, w_temp , r_temp, 1000, 1e-6, max_iter_price, tol_price, 1)
    end
println("Time taken for 6.3: $time_taken seconds")


# Compute the consumption policy function
C_policy = compute_consumption_policy(params, L_policy, I_policy, w_func, r_func)

# Interpolate all policy functions
C_policy_interp = LinearInterpolation((τ_states, z_states, I_grid, K_grid), C_policy, extrapolation_bc=Line())
L_policy_interp = LinearInterpolation((τ_states, z_states, I_grid, K_grid), L_policy, extrapolation_bc=Line())
I_policy_interp = LinearInterpolation((τ_states, z_states, I_grid, K_grid), I_policy, extrapolation_bc=Line())
w_func_interp = LinearInterpolation((τ_states, z_states, I_grid, K_grid), w_func, extrapolation_bc=Line())
r_func_interp = LinearInterpolation((τ_states, z_states, I_grid, K_grid), r_func, extrapolation_bc=Line())

# Simulate the IRFs
# Generate IRFs for a tax shock
T = 20  # Time horizon
C_irf_tax, L_irf_tax, I_irf_tax, K_irf_tax, w_irf_tax, r_irf_tax = simulate_irf(params, :tax, T, C_policy_interp, L_policy_interp, I_policy_interp, w_func_interp, r_func_interp)

# Prepare data for plotting
time = 1:T

# Create a figure with 3 rows and 2 columns
plot_layout = @layout [a b; c d; e f]
p1 = plot(layout = plot_layout, size=(800, 600))

# Plot each variable's IRF in a subplot
plot!(p1[1], time, C_irf_tax, label=nothing, title="Consumption", xlabel="Time", ylabel="Deviation")
plot!(p1[2], time, L_irf_tax, label=nothing, title="Labor", xlabel="Time", ylabel="Deviation")
plot!(p1[3], time, I_irf_tax, label=nothing, title="Investment", xlabel="Time", ylabel="Deviation")
plot!(p1[4], time, K_irf_tax, label=nothing, title="Capital", xlabel="Time", ylabel="Deviation")
plot!(p1[5], time, w_irf_tax, label=nothing, title="Wage", xlabel="Time", ylabel="Deviation")
plot!(p1[6], time, r_irf_tax, label=nothing, title="Interest Rate", xlabel="Time", ylabel="Deviation")

# Display the figure
display(p1)
savefig("IRF_tax.png")

# Generate IRFs for a technology shock
T = 20  # Time horizon
C_irf_tech, L_irf_tech, I_irf_tech, K_irf_tech, w_irf_tech, r_irf_tech = simulate_irf(params, :technology, T, C_policy_interp, L_policy_interp, I_policy_interp, w_func_interp, r_func_interp)

# Prepare data for plotting
time = 1:T

# Create a figure with 3 rows and 2 columns
plot_layout = @layout [a b; c d; e f]
p2 = plot(layout = plot_layout, size=(800, 600))

# Plot each variable's IRF in a subplot
plot!(p2[1], time, C_irf_tech, label=nothing, title="Consumption", xlabel="Time", ylabel="Deviation")
plot!(p2[2], time, L_irf_tech, label=nothing, title="Labor", xlabel="Time", ylabel="Deviation")
plot!(p2[3], time, I_irf_tech, label=nothing, title="Investment", xlabel="Time", ylabel="Deviation")
plot!(p2[4], time, K_irf_tech, label=nothing, title="Capital", xlabel="Time", ylabel="Deviation")
plot!(p2[5], time, w_irf_tech, label=nothing, title="Wage", xlabel="Time", ylabel="Deviation")
plot!(p2[6], time, r_irf_tech, label=nothing, title="Interest Rate", xlabel="Time", ylabel="Deviation")

# Display the figure
display(p2)
savefig("IRF_tech.png")


"""
6.6 Switching between VFI and PFI
"""
num_K = 250 
num_I = 50
K_grid = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K))
I_grid = collect(range(0.5 * I_ss, 1.5 * I_ss, length=num_I))

# Initialize parameters struct
params = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)

# Intialize guess for V, w, r
V_guess = ones(num_τ, num_z, num_I, num_K) .* (log(C_ss) + 0.2 * log(G_ss) - 0.5 * L_ss^2)
w_guess = ones(num_τ, num_z, num_I, num_K) .* w_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)
r_guess = ones(num_τ, num_z, num_I, num_K) .* r_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)

# Set max iterations and tolerance of price
max_iter_price = 10
tol_price = 1e-3
time_taken = @elapsed begin
# First find the Eq price with max_iter_V = 50 and tol_V = 1e-3
V_temp, L_policy_temp, I_policy_temp, K_policy_temp, w_temp , r_temp = main(params, V_guess, w_guess, r_guess, 50, 1e-3, max_iter_price, tol_price, 2)
# Then find the associated value function with max_iter_V = 1000 and tol_V = 1e-6
V, L_policy, I_policy, K_policy, w , r = main(params, V_temp, w_temp , r_temp, 1000, 1e-6, max_iter_price, tol_price, 2)
end
println("Time taken for 6.6: $time_taken seconds")



"""
6.7 Mutigrid
"""
# Generate I grids (fixed)
num_I = 50
I_grid = collect(range(0.5 * I_ss, 1.5 * I_ss, length=num_I))

## 100 points
# Generate K grids with 100 points
num_K = 100 
K_grid = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K))

# Initialize parameters struct
params = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)

# Intialize guess for V, w, r
V_guess = ones(num_τ, num_z, num_I, num_K) .* (log(C_ss) + 0.2 * log(G_ss) - 0.5 * L_ss^2)
w_guess = ones(num_τ, num_z, num_I, num_K) .* w_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)
r_guess = ones(num_τ, num_z, num_I, num_K) .* r_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)

# VFI with 100 grids: max_iter_price = 20, tol_price = 1e-3, max_iter_V = 100 and tol_V = 1e-4
time_taken_1 = @elapsed begin
V_1, L_policy_1, I_policy_1, K_policy_1, w_1 , r_1 = main(params, V_guess, w_guess, r_guess, 100, 1e-4, 10, 1e-3, 1)
end
println("Time taken for 6.7 step 1: $time_taken_1 seconds")


## 500 points
# Generate K grids with 500 points
num_K_fine = 500 
K_grid_fine = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K_fine))

# Initialize parameters struct
params_fine = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)
# Interpolate the value and pricing functions to form the intial guessses
V_interp = Array{Any}(undef, num_τ, num_z)
w_interp = Array{Any}(undef, num_τ, num_z)
r_interp = Array{Any}(undef, num_τ, num_z)
V_fine_guess = ones(num_τ, num_z, num_I, num_K_fine)
w_fine_guess = ones(num_τ, num_z, num_I, num_K_fine) 
r_fine_guess = ones(num_τ, num_z, num_I, num_K_fine)
for τ_idx = 1:num_τ
    for z_idx = 1:num_z
        V_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid), V_1[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        w_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid), w_1[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        r_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid), r_1[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        for I_idx = 1:num_I
            for K_idx = 1:num_K_fine
                V_fine_guess[τ_idx, z_idx, I_idx, K_idx] = V_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_fine[K_idx])
                w_fine_guess[τ_idx, z_idx, I_idx, K_idx] = w_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_fine[K_idx])
                r_fine_guess[τ_idx, z_idx, I_idx, K_idx] = r_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_fine[K_idx])
            end
        end
    end
end

# VFI with 500 grids: max_iter_price = 20, tol_price = 1e-3, max_iter_V = 500 and tol_V = 1e-5
time_taken_2 = @elapsed begin
    V_2, L_policy_2, I_policy_2, K_policy_2, w_2 , r_2 = main(params_fine, V_fine_guess, w_fine_guess, r_fine_guess, 500, 1e-5, 10, 1e-3, 1)
    end
println("Time taken for 6.7 step 2: $time_taken_2 seconds")


## 5000 points
# Generate K grids with 5000 points
num_K_final = 5000
K_grid_final = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K_final))

# Initialize parameters struct
params_final = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)
# Interpolate the value and pricing functions to form the intial guessses
V_interp = Array{Any}(undef, num_τ, num_z)
w_interp = Array{Any}(undef, num_τ, num_z)
r_interp = Array{Any}(undef, num_τ, num_z)
V_final_guess = ones(num_τ, num_z, num_I, num_K_final)
w_final_guess = ones(num_τ, num_z, num_I, num_K_final) 
r_final_guess = ones(num_τ, num_z, num_I, num_K_final)
for τ_idx = 1:num_τ
    for z_idx = 1:num_z
        V_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid_fine), V_2[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        w_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid_fine), w_2[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        r_interp[τ_idx, z_idx] = LinearInterpolation((I_grid, K_grid_fine), r_2[τ_idx, z_idx, :, :], extrapolation_bc=Line())
        for I_idx = 1:num_I
            for K_idx = 1:num_K_final
                V_final_guess[τ_idx, z_idx, I_idx, K_idx] = V_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_final[K_idx])
                w_final_guess[τ_idx, z_idx, I_idx, K_idx] = w_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_final[K_idx])
                r_final_guess[τ_idx, z_idx, I_idx, K_idx] = r_interp[τ_idx, z_idx](I_grid[I_idx], K_grid_final[K_idx])
            end
        end
    end
end

# VFI with 5000 grids: max_iter_price = 20, tol_price = 1e-3, max_iter_V = 1000 and tol_V = 1e-6
time_taken_3 = @elapsed begin
    V_3, L_policy_3, I_policy_3, K_policy_3, w_3 , r_3 = main(params_final, V_final_guess, w_final_guess, r_final_guess, 1000, 1e-6, 10, 1e-3, 1)
    end
println("Time taken for 6.7 step 3: $time_taken_3 seconds")

total_time = time_taken_1 + time_taken_2 + time_taken_3
println("Time taken for 6.7 in total: $total_time seconds")



"""
6.8 Stochastic grid
"""
include("q6_ge_func_stochastic.jl") #exclude "q6_ge_func.jl" at the beginning
Random.seed!(1234)
num_K = 500 
num_I = 50
K_grid = collect(range(0.7 * K_ss, 1.3 * K_ss, length=num_K))
I_grid = collect(range(0.5 * I_ss, 1.5 * I_ss, length=num_I))
# Initialize parameters struct
params = params_final = ModelParams(β, α, δ, ϕ, τ_states, π_τ, z_states, π_z, μ_τ, ρ_τ, σ_τ, μ_z, ρ_z, σ_z, K_ss, K_grid, I_grid, num_K, num_I, num_τ, num_z)

# Intialize guess for V, w, r
V_guess = ones(num_τ, num_z, num_I, num_K) .* (log(C_ss) - 0.5 * L_ss^2)
w_guess = ones(num_τ, num_z, num_I, num_K) .* w_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)
r_guess = ones(num_τ, num_z, num_I, num_K) .* r_ss .* reshape(exp.(z_states), 1, num_z, 1, 1)

# Set max iterations and tolerance of price
max_iter_price = 10
tol_price = 1e-3
time_taken = @elapsed begin
# First find the Eq price with max_iter_V = 50 and tol_V = 1e-3
V_temp, L_policy_temp, I_policy_temp, K_policy_temp, w_temp , r_temp, K_grid_gen_temp = main!(params, V_guess, w_guess, r_guess, 50, 1e-3, max_iter_price, tol_price, 3)
# Then find the associated value function with max_iter_V = 1000 and tol_V = 1e-6
V, L_policy, I_policy, K_policy, w , r, K_grid_gen = main!(params, V_temp, w_temp , r_temp, 1000, 1e-6, max_iter_price, tol_price, 3)
end
println("Time taken for 6.8: $time_taken seconds")
