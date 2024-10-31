using NLsolve
using Random


# Optimal consumption based on FOC and budget constraint
function optimal_consumption(p, e, α, ω)
    m, n = size(e)  # m goods, n agents
    x = zeros(n, m)

    # For each agent, solve for optimal consumption
    for j in 1:n
        budget = sum(p .* e[j, :])  # Total wealth of agent j
        # Solve for the Lagrangian
        Lagrangian_solve(t) = any(x -> x <= 0, p * t[1]) ? 10^8 : sum(p[i] * (p[i] * t[1] / α[j, i])^(1 / ω[j, i]) for i in 1:m) - budget
        μ0 = [1.0]
        result = nlsolve(Lagrangian_solve, μ0, autodiff = :forward)
        μ = result.zero[1]

        for i in 1:m
            # Use FOC to determine x_j^i (optimal consumption)
            x[j, i] = p[i]*μ<=0 ? 10^8 : (p[i] * μ / α[j, i])^(1 / ω[j, i])
        end
    end
    return x
end

# Excess demand vector
function excess_demand(p, e, α, ω)
    m, n = size(e)
    x = optimal_consumption(p, e, α, ω)
    z = zeros(m)

    # Calculate excess demand for each good i
    for i in 1:m
        total_consumption = sum(x[:, i])  # Total consumption of good i across agents
        total_endowment = sum(e[:, i])  # Total endowment of good i across agents
        z[i] = total_consumption - total_endowment
    end
    
    return z
end

# Example for m = n = 10
m, n = 10, 10  # 10 goods, 10 agents

# Generate parameters and endowments
e = rand(5.0:0.5:20.0, n, m)
# α = fill(1.0, n, m)  # All α's set to 1.0
α = rand(0.5:0.1:1.5, n, m)
ω = rand([-2, -1, -0.5], n, m)
#ω = fill(-0.5, n, m) # All ω's set to -0.5

# Initial guess for prices
p0 = 0.5*ones(m) 

# Solve for equilibrium prices using NLsolve
result = nlsolve(p->excess_demand(p, e, α, ω), p0)

# Extract equilibrium prices
p_eq = result.zero
# Set good 1 as numeraire
p_eq = p_eq * 1.0 / p_eq[1]

# Output equilibrium prices
println("Equilibrium prices: ", p_eq)
