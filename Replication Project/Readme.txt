This file contains the replication files for Section 6 of "Heterogeneity and Aggregate Fluctuations" by Minsu Chang, Xiaohong Chen, and Frank Schorfheide (CCS thereafter).

Main folders:
"Data" contains the empirical data provided by CCS
"Main_Code" contains the main Julia files that generate the results.
"Functions" contains the Julia functions associated with "Main_Code".
"SpecFiles"  contains files with information on the model specifications and MCMC configurations.
"Graph_Code" contains the Matlab files that plot the replication results.
"Figures" contains the replication results.


"Main_Code":
Aggregate VAR: AggVAR_MDD.jl, AggVAR_Estimation.jl, AggVAR_IRF.jl (run the scripts in this order)

Inequality VAR: AltVAR_MDD.jl, AltVAR_Estimation.jl, AltVAR_IRF.jl (run the scripts in this order)

Funcitional state space model: 
1. FSS_Density_Estimation.j: estimates log spline densities by MLE.
2. FSS_PredPercentiles_MLE.jl: computes percentiles based on estimated log spline densities.
3. FSS_MDD.jl: loops over K, lambda, and number of lags to compute the MDD.
4. FSS_SS_Estimation.jl: estimates state-space representation conditional on K and lambda. 
5. FSS_PredPercentiles_Alpha_Smoothed.jl: computes percentiles from smoothed states using posterior mean parameter estimates.
6. FSS_IRF.jl: creates IRFs to the three aggregate shocks identified by Cholesky factorization.
7. FSS_IRF_DistrSh.jl:  generates a variety of IRF plots for distributional shocks.
8. FSS_IRF_Transform.jl:  compute IRFs for various transformations of the original density responses.


"Graph_Code":
1. Graph_Densities.m: overlays estimated densities for different K and histograms.
2. Graph_Alpha_Smoothed.m: overlays actual a-hat's and smoothed a's.
3. Graph_Percentiles_MLE.m: generates figure of fitted versus empirical percentiles.
4. Graph_IRF_fVAR_AggV_AggSh.m: generates IRFs of aggregate variables to aggregate shocks. (Aggregate VAR vs. Functional model)
5. Graph_IRF_fVAR_AggSh.m: generates IRFs of inequality measures to aggregate shocks.
6. Graph_IRF_fVAR_AggSh_nozeros.m: generates IRFs of inequality measures (no pointmass at 0) to aggregate shocks.
7. Graph_IRF_fVAR_DistrSh.m: generates IRFs of aggregate variables to distributional shocks.
8. Graph_IRF_fVAR_v_AltVAR.m: generates IRFs of the functional model and inequality VAR.

