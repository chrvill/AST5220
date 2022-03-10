# AST5220

Code that will eventually produce the CMB power spectrum. So far the code only computes the following:

  - Background cosmology parameters (eta, H, Hp, t, + derivatives of H and Hp)
  - Recombination history related parameters (X_e, n_e, tau, g_tilde)

Input parameters:
  - h, OmegaB, OmegaCDM, OmegaK, Neff, TCMB for BackgroundCosmology
  - Yp (+ data from BackgroundCosmology object) for RecombinationHistory

Compiling:
make

Running C++ computations:
./cmb

Plotting results:
python3 plot.py
