This folder contains the code for ECON 8210 HW2.

1. Folder "Projection": 
Chebyshev.m is the main code that uses the projection method to compute the policy functions and Euler errors.

Functions 
(1) tauchen.m: approximates AR(1) process.
(2) rando.m: generates a random variable in 1, 2, ..., n given a distribution vector. 
(3) steady_state_system.m: solves for steady state. 
(4) residual_fcn.m: solves for the coefficients associated to the Chebyshev polynomials.
(5) eulerr_single.m: computes Euler error for a given capital and productivity level.
(6) eulerr_grid.m: computes Euler errors on a grid for capital and productivity level.
(7) simulation.m: simulates the economy.


2. Folder "Finite element": 
fe.m is the main code that uses the finite element method to compute the policy functions and Euler errors.

Functions 
(1) tauchen.m: approximates AR(1) process.
(2) lgwt.m: computes the Legendre-Gauss nodes and weights on the interval [A,B].
(3) steady_state_system.m: solves for steady state. 
(4) basisFunctions.m:  evaluates the piecewise linear basis functions.
(5) interval_support.m: returns the interval support of piecewise linear basis functions.
(6) lOfkz.m: computes the labor supply level given basis functions and their coefficients.
(7) computeResidual.m: compute the Euler residual for a given capital and productivity level.
(8) residualSystem.m: solves for the coefficients associated to the FE basis functions.
(9) eer.m: computes Euler error for a given capital and productivity level.


3. Folder "Perturbation": 
perturbation.mod solves the model using third-order perturbation in Dynare.
perturbation_policy.m plots the policy functions.
perturbation_eer.m plots the Euler errors.

Functions 
(1) tauchen.m: approximates AR(1) process.
(2) gaussHermiteQuadrature.m: computes the nodes and weights for the Gauss-Hermite quadrature.
(3) compute_policy_functions.m: backs out the policy functions from perturbation coefficients.


4. Folder "DNN": 
DNN.ipynb is the main code that uses deep neural networks to compute the policy functions and Euler errors.
