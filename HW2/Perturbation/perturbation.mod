// Solve neoclassical growth model using third-order perturbation in dynare

var c l k z;          // Endogenous variables: consumption, labor, capital, technology level
varexo e;             // Exogenous shock

parameters beta alpha delta lambda sigma; 
// Parameter values
beta   = 0.97;
alpha  = 0.33;
delta  = 0.1;
lambda = 0.95;
sigma  = 0.007;

// Model block
model;
    // From the FOCs and constraints:
    // 1) Euler Equation:
    (1/c) = beta*((alpha*exp(z(+1))*(k)^(alpha-1)*l(+1)^(1 - alpha) + 1 - delta)*(1/c(+1)));

    // 2) Labor condition:
    c = (1 - alpha)*exp(z)*k(-1)^(alpha)*l^(- alpha - 1);

    // 3) Resource constraint:
    c + k - (exp(z)*k(-1)^(alpha)*l^(1 - alpha) + (1 - delta)*k(-1)) = 0;

    // 4) AR(1) process for z:
    z = lambda*z(-1) + sigma*e;
end;


// Initial values
initval;
    z = 0;
    k = 3.7; 
    c = 1; 
    l = 0.9;
    e = 0;
end;

// Steady state
steady;

// Eigenvalues
check;

// Simulate the model
stoch_simul(order=3, irf=20);
