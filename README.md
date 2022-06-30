# AST5220

Code for reproducing the CMB power spectrum. It computes the following:

  - Background cosmology parameters (eta, H, Hp, t, + derivatives of H and Hp)
  - Recombination history related parameters (X_e, n_e, tau, g_tilde)
  - Evolution of perturbations (delta_cdm, delta_b, v_cdm, v_b, theta_0, theta_1, etc.)
  - CMB power spectrum (C_ells) and matter power spectrum

Input parameters:
  - h, OmegaB, OmegaCDM, OmegaK, Neff, TCMB for BackgroundCosmology
  - Yp (+ data from BackgroundCosmology object) for RecombinationHistory
  - A_s, n_s and k_pivot (+ data from previous milestones) for PowerSpectrum.

In the perturbation-code I have implemented the possibility of including neutrinos and polarization. All the results in the report are, however, produced with neutrinos and polarization turned off. I mostly included it in the code because it wasn't that much more work and makes the code more complete. But I don't feel like trying to explain
neutrinos or polarization in the report, which is why I turned them off. The C++ computations run in about 20-30 seconds
on my laptop.

Compiling:
make

Running C++ computations:
./cmb

Plotting results:
python3 plot.py

![plot](./images/cmb_map.pdf)
