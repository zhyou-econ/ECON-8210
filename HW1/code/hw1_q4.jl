using Optim
using Random
using DataFrames

Random.seed!(1234)

# Function to generate heterogeneous α, ω, and λ matrices for m = n = 10
function generate_hetero_parameters(m, n)
    α = rand(0.5:0.1:1.5, n, m)  # α in [0.5, 1.5]
    ω = rand(-0.6:0.05:-0.2, n, m)  # ω in [-0.6, -0.2]
    λ = rand(0.5:0.1:1.5, n)  # λ in [0.5, 1.5]
    λ = λ * (n / sum(λ))
    return α, ω, λ
end

# Define the utility function for each agent
function utility(x, α, ω)
    m = length(x)
    return sum(α[i] * x[i]^(1 + ω[i]) / (1 + ω[i]) for i in 1:m)
end

# Define the social planner's objective function (to maximize social welfare)
function social_welfare(x, α, ω, λ, e)
    n = length(λ)  # Number of agents
    m = div(length(x), n)  # Number of goods per agent
    total_utility = 0.0
    for j in 1:n
        allocation = x[1 + (j - 1) * m:j * m]  # Allocation of goods for agent j
        total_utility += λ[j] * utility(allocation, α[j, :], ω[j, :])
    end

    # Penalize for exceeding endowments
    penalty = 0.0
    for i in 1:m
        allocation_i_sum = sum(x[i:m:i + (n - 1) * m])
        endowment_i_sum = sum(e[:,i])
        penalty += max(0, allocation_i_sum - endowment_i_sum)^2  # Penalty if allocation exceeds endowment
    end
    
    return -(total_utility - 10^8 * penalty)  # Negate because we're minimizing
end


m = 10  # Number of goods
n = 10  # Number of agents
e = rand(10.0:1.0:20.0, n, m)  # Endowments for each agent
x0 = reshape(e, (n*m , 1))


# Case 1: Homogeneous parameters
# Parameters
α = fill(1.0, n, m)  # All α's set to 1.0
ω = fill(-0.5, n, m)  # All ω's set to -0.5
λ = fill(1.0, n)  # All social weights set to 1.0

# Define the objective function
f = x -> social_welfare(x, α, ω, λ, e)

# Run Nelder-Mead optimization (without gradients or Jacobians)
result = optimize(f, x0, NelderMead(), Optim.Options(iterations=1000))

println("Optimal allocation (homogeneous): ", result.minimizer)
println("Maximized social welfare (homogeneous): ", -result.minimum)

# Compute each agent's utility
optimal_allocation_homogeneous = result.minimizer
utilities_homogeneous = []
for j in 1:n
    allocation = optimal_allocation_homogeneous[1 + (j - 1) * m:j * m]
    push!(utilities_homogeneous, utility(allocation, α[j, :], ω[j, :]))
end


# Case 2: Heterogeneous parameters
α_hetero, ω_hetero, λ_hetero = generate_hetero_parameters(m, n)
#α_hetero = α
#ω_hetero = ω

# Objective function for the heterogeneous case
f_hetero = x -> social_welfare(x, α_hetero, ω_hetero, λ_hetero, e)

# Run Nelder-Mead optimization for the heterogeneous case
result_hetero = optimize(f_hetero, x0, NelderMead(), Optim.Options(iterations=1000))

println("Optimal allocation (heterogeneous): ", result_hetero.minimizer)
println("Maximized social welfare (heterogeneous): ", -result_hetero.minimum)

# Calculate each individual's utility
optimal_allocation_heterogeneous = result.minimizer
utilities_heterogeneous = []
for j in 1:n
    allocation = optimal_allocation_heterogeneous[1 + (j - 1) * m:j * m]
    push!(utilities_heterogeneous, utility(allocation, α_hetero[j, :], ω_hetero[j, :]))
end

# Generate table for comparison
df = DataFrame(
    agent = 1:n, 
    λ_homogeneous = λ, 
    utility_homogeneous = utilities_homogeneous, 
    λ_heterogeneous = λ_hetero, 
    utility_heterogeneous = utilities_heterogeneous
)

println(df)


