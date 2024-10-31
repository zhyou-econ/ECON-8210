using LinearAlgebra

# Newton Raphson
function newton_raphson(f, grad_f, hessian_f, x0; tol=1e-5, max_iter=10000)
    x = x0
    for i in 1:max_iter
        g = grad_f(x)
        H = hessian_f(x)
        delta_x = -H \ g  # Solve for Newton step
        x = x + delta_x
        if norm(delta_x) < tol
            return x, i
        end
    end
    return x, max_iter
end


# BFGS
function bfgs(f, grad_f, x0; tol=1e-5, max_iter=10000)
    x = x0
    B = I  # Initial inverse Hessian as identity matrix
    for i in 1:max_iter
        g = grad_f(x)
        if norm(g) < tol
            return x, i
        end
        x_new = x - B * g
        s = x_new - x
        y = grad_f(x_new) - g
        B = (I(size(g, 1)) - (y' * s)\(s * y')) * B * (I(size(g, 1)) - (y' * s)\(y * s')) + (y' * s)\(s * s')
        x = x_new
    end
    return x, max_iter
end


# Steepest descent
function steepest_descent(f, grad_f, x0; tol=1e-5, max_iter=10000)
    x = x0
    iter = 0

    while norm(grad_f(x)) > tol && iter < max_iter
        # Compute gradient at current point
        g = grad_f(x)
        # Find optimal alpha using line search
        α = line_search_steep(f, x, -g)
        # Update position
        x = x - α * g
        iter += 1
    end

    return x, iter
end

# Line search function to find optimal step size α
function line_search_steep(f, x, d; α=1.0, β=0.5, γ=0.1)
    while f(x + α * d) > f(x) + γ * α * dot(grad_rosenbrock(x), d)
        α *= β  # reduce step size
    end
    return α
end


# Conjugate gradient
function conjugate_gradient(f, grad_f, x0; tol=1e-5, max_iter=10000)
    x = x0
    g = grad_f(x)
    d = -g
    iter = 0

    while norm(g) > tol && iter < max_iter
        # Step size (using line search)
        α = line_search_conjugate(f, x, d)
        # Update position
        x_new = x + α * d
        # Compute new gradient
        g_new = grad_f([x_new[1], x_new[2]])
        # Compute beta using Fletcher-Reeves formula
        β = dot(g_new, g_new) / dot(g, g)
        # Update direction
        d = -g_new + β * d
        # Prepare for next iteration
        x = x_new
        g = g_new
        iter += 1
    end

    return x, iter

end

# Line search function to find optimal step size α
function line_search_conjugate(f, x, d; α=1.0, β=0.5, γ=0.1)
    while f([x[1] + α * d[1], x[2] + α * d[2]]) > f([x[1], x[2]]) + γ * α * dot(grad_rosenbrock([x[1], x[2]]), d)
        α *= β  # reduce step size
    end
    return α
end


# Rosenbrock function, gradient, and Hessian
function f_rosenbrock(x)
    return 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end

function grad_rosenbrock(x)
    return [
        400 * x[1] * (x[1]^2 - x[2]) + 2 * (x[1] - 1);
        200 * (x[2] - x[1]^2)
    ]
end

function hessian_rosenbrock(x)
    return [
        1200 * x[1]^2 - 400 * x[2] + 2  -400 * x[1];
        -400 * x[1]  200
    ]
end


# Initial guess
x0 = [0, 0]

# Measure time and iterations for each method
println("Comparing performance:")
@time x_newton, iter_newton = newton_raphson(f_rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)
@time x_bfgs, iter_bfgs = bfgs(f_rosenbrock, grad_rosenbrock, x0)
@time x_sd, iter_sd = steepest_descent(f_rosenbrock, grad_rosenbrock, x0)
@time x_cg, iter_cg = conjugate_gradient(f_rosenbrock, grad_rosenbrock, x0)

# Print results in a table format
println("Results:")
println("Newton-Raphson: $x_newton")
println("BFGS: $x_bfgs")
println("Steepest Descent: $x_sd")
println("Conjugate Gradient: $x_cg")

# Print time in a table format
println("\nMethod Comparison:")
println("--------------------------------------------------")
println("| Method           | Time (seconds) | Iterations |")
println("--------------------------------------------------")
@time begin
    println("| Newton-Raphson   | ", @elapsed(newton_raphson(f_rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)), "  | ", iter_newton, "        |")
    println("| BFGS             | ", @elapsed(bfgs(f_rosenbrock, grad_rosenbrock, x0)), "  | ", iter_bfgs, "        |")
    println("| Steepest Descent | ", @elapsed(steepest_descent(f_rosenbrock, grad_rosenbrock, x0)), "  | ", iter_sd, "        |")
    println("| Conjugate Gradient | ", @elapsed(conjugate_gradient(f_rosenbrock, grad_rosenbrock, x0)), "  | ", iter_cg, "        |")
end
println("--------------------------------------------------")
