using Random, Plots, CSV, DataFrames, LaTeXStrings

# Define the parameters
T = 100.0
rho = 0.04
lambda = 0.02

# Define the function u(c) = -e^(-c)
u(c) = -exp(-c)

# Define the integrand
f(t) = exp(-rho * t) * u(1 - exp(-lambda * t))

# Midpoint rule
function midpoint_rule(f, a, b, n)
    h = (b - a) / n
    sum = 0.0
    for i in 1:n
        t = a + (i - 0.5) * h
        sum += f(t)
    end
    return h * sum
end

# Trapezoidal rule
function trapezoidal_rule(f, a, b, n)
    h = (b - a) / n
    sum = 0.5 * (f(a) + f(b))
    for i in 1:(n-1)
        t = a + i * h
        sum += f(t)
    end
    return h * sum
end

# Simpson's rule
function simpson_rule(f, a, b, n)
    if n % 2 != 0
        error("Simpson's rule requires an even number of intervals")
    end
    h = (b - a) / n
    sum = f(a) + f(b)
    for i in 1:2:(n-1)
        t = a + i * h
        sum += 4 * f(t)
    end
    for i in 2:2:(n-2)
        t = a + i * h
        sum += 2 * f(t)
    end
    return (h / 3) * sum
end


# Monte Carlo integration
function monte_carlo(f, a, b, n)
    # Calculate width of each subinterval
    interval_width = (b - a) / n
    # Initialize the sum for the Monte Carlo estimate
    integral_sum = 0.0
    
    # Perform the Monte Carlo integration
    for i in 1:n
        # Determine the start of the current subinterval
        t_start = a + (i - 1) * interval_width
        # Draw a sample from the subinterval uniformly
        t_sample = t_start + rand() * interval_width
        # Evaluate f(t_sample) and add it to the integral sum
        integral_sum += f(t_sample)
    end
    
    # Multiply by interval width to get the estimate of the integral
    return integral_sum * interval_width
end

# Generate a reference value using a very fine discretization (10^8 intervals)
n_ref = 10^8
reference_value = simpson_rule(f, 0, T, n_ref)

# Function to compute relative error
relative_error(computed_value, reference_value) = abs(computed_value - reference_value) / abs(reference_value)

# Varying number of intervals
n_values = [10, 100, 1000, 10^4, 10^5, 10^6]

# Arrays to store errors and computation times
midpoint_errors, trapezoidal_errors, simpson_errors, monte_carlo_errors = [], [], [], []
midpoint_times, trapezoidal_times, simpson_times, monte_carlo_times = [], [], [], []

for n in n_values
    # Midpoint Rule
    mid_time = @elapsed mid_res = midpoint_rule(f, 0, T, n)
    push!(midpoint_errors, relative_error(mid_res, reference_value))
    push!(midpoint_times, mid_time)
    
    # Trapezoidal Rule
    trap_time = @elapsed trap_res = trapezoidal_rule(f, 0, T, n)
    push!(trapezoidal_errors, relative_error(trap_res, reference_value))
    push!(trapezoidal_times, trap_time)
    
    # Simpson's Rule
    simp_time = @elapsed simp_res = simpson_rule(f, 0, T, n)
    push!(simpson_errors, relative_error(simp_res, reference_value))
    push!(simpson_times, simp_time)
    
    # Monte Carlo Method
    mc_time = @elapsed mc_res = monte_carlo(f, 0, T, n)
    push!(monte_carlo_errors, relative_error(mc_res, reference_value))
    push!(monte_carlo_times, mc_time)
end

# Create a table of results (errors and times)
table_data = hcat(n_values, midpoint_errors, trapezoidal_errors, simpson_errors, monte_carlo_errors,
                  midpoint_times, trapezoidal_times, simpson_times, monte_carlo_times)
header = ["n", "Midpoint Error", "Trapezoidal Error", "Simpson Error", "Monte Carlo Error",
"Midpoint Time", "Trapezoidal Time", "Simpson Time", "Monte Carlo Time"];
CSV.write("C:/Users/Howard You/Desktop/Penn courses/ECON 8210 Quantitative Macro (Jesus)/Homework/HW1/Q2_Table.csv", DataFrame(table_data, :auto), header=header) 

# Plot the relative errors
plot(n_values, midpoint_errors, label="Midpoint Rule", lw=2, marker=:o, yscale=:log10, xscale=:log10, legend=:right)
plot!(n_values, trapezoidal_errors, label="Trapezoidal Rule", lw=2, marker=:o)
plot!(n_values, simpson_errors, label="Simpson's Rule", lw=2, marker=:o)
plot!(n_values, monte_carlo_errors, label="Monte Carlo", lw=2, marker=:o)
xticks!([10, 100, 10^3, 10^4, 10^5, 10^6], ["10", "100", L"10^3", L"10^4", L"10^5", L"10^6"])
xlabel!("Number of Intervals (n)")
ylabel!("Relative Error")
title!("Comparison of Relative Errors")
savefig("C:/Users/Howard You/Desktop/Penn courses/ECON 8210 Quantitative Macro (Jesus)/Homework/HW1/Q2_Relative_Error_Plot.pdf")

# Plot the computation times
plot(n_values, midpoint_times, label="Midpoint Rule", lw=2, marker=:o, yscale=:log10, xscale=:log10, legend=:bottomright)
plot!(n_values, trapezoidal_times, label="Trapezoidal Rule", lw=2, marker=:o)
plot!(n_values, simpson_times, label="Simpson's Rule", lw=2, marker=:o)
plot!(n_values, monte_carlo_times, label="Monte Carlo", lw=2, marker=:o)
xticks!([10, 100, 10^3, 10^4, 10^5, 10^6], ["10", "100", L"10^3", L"10^4", L"10^5", L"10^6"])
xlabel!("Number of Intervals (n)")
ylabel!("Computation Time (seconds)")
title!("Comparison of Computation Time")
savefig("C:/Users/Howard You/Desktop/Penn courses/ECON 8210 Quantitative Macro (Jesus)/Homework/HW1/Q2_Computation_Time_Plot.pdf")
