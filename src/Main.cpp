#include "Utils.h"
#include "BackgroundCosmology.h"
#include "RecombinationHistory.h"
#include "Perturbations.h"
#include "PowerSpectrum.h"

int main(int argc, char **argv){
  Utils::StartTiming("Everything");

  std::string txt_prefix = "txtfiles/";

  //=========================================================================
  // Parameters
  //=========================================================================

  // Background parameters
  double h           = 0.67;
  double OmegaB      = 0.05;
  double OmegaCDM    = 0.267;
  double OmegaK      = 0.0;
  double Neff        = 0.0;
  double TCMB        = 2.7255;

  double Yp          = 0.245;

  // Power-spectrum parameters
  double A_s         = 2.1e-9;
  double n_s         = 0.965;
  double kpivot_mpc  = 0.05;

  //=================================================
  // Runs where I change the cosmological parameters
  //=================================================
  /*
  // Changing OmegaLambda. Then OmegaB*h^2 = x_b and OmegaCDM*h^2 = x_cdm should be kept constant.
  double x_b    = OmegaB*h*h;
  double x_cdm  = OmegaCDM*h*h;
  double x      = x_b + x_cdm;

  double OmegaLambda = 0.9;
  // h needs to change if OmegaB + OmegaCDM + OmegaLambda = 1, while OmegaB*h^2 etc. are constant.
  h = std::sqrt(x/(1.0 - OmegaLambda));
  // New values for OmegaB and OmegaCDM with OmegaLambda = 0.9
  OmegaB = x_b/(h*h);
  OmegaCDM = x_cdm/(h*h);
  */
  /*
  // Changing OmegaM. Then want OmegaLambda*h^2 and OmegaB*h^2 to be constant.
  double OmegaLambda = 0.683; // Fiducial value
  double x_b      = OmegaB*h*h;
  double x_lambda = OmegaLambda*h*h;

  double OmegaM = 0.5;
  h = std::sqrt(x_lambda/(1.0 - OmegaM));

  OmegaB = x_b/(h*h);
  OmegaCDM = OmegaM - OmegaB;
  */
  /*
  // Changing OmegaB. Want OmegaM to be constant (and thus OmegaLambda is also constant)
  double OmegaM = OmegaB + OmegaCDM;
  OmegaB = 0.1;
  OmegaCDM = OmegaM - OmegaB;
  */

  //=========================================================================
  // Module I
  //=========================================================================

  // Set up and solve the background
  BackgroundCosmology cosmo(h, OmegaB, OmegaCDM, OmegaK, Neff, TCMB);
  //cosmo.set_xlims(-20, 5); // Setting the x-range
  cosmo.solve();
  cosmo.info();

  double age = cosmo.age_of_universe(0);

  std::cout << "Age of the universe: " << age << " Gyr\n";

  // Defining x-array that is used when computing luminosity distances
  // that are compared with observed lum. distances
  Vector x_array = Utils::linspace(-0.85, 0, 100);
  std::ofstream d_L{txt_prefix + "lum_dist.txt"};

  // Writing x and lum. distance to file for each value of x
  for (int i = 0; i < x_array.size(); ++i)
  {
    d_L << x_array[i] << "\t" << cosmo.d_L(x_array[i]) << "\n";
  }


  // Output background evolution quantities
  cosmo.output(txt_prefix + "cosmology.txt");

  //=========================================================================
  // Module II
  //=========================================================================

  // Solve the recombination history
  RecombinationHistory rec(&cosmo, Yp);
  rec.solve();
  rec.info();

  std::cout << std::scientific;
  std::cout << "Xe today: " << rec.Xe_of_x(0) << "\n\n";

  // Output recombination quantities
  rec.output(txt_prefix + "recombination.txt");

  //=========================================================================
  // Module III
  //=========================================================================

  // Solve the perturbations
  Perturbations pert(&cosmo, &rec);
  pert.info();

  Vector kvalues{0.3/Constants.Mpc, 0.05/Constants.Mpc, 0.0001/Constants.Mpc};

  pert.solve();

  // Computing evolution of perturbations for the different values of k
  // in the kvalues-vector
  for (int i = 0; i < kvalues.size(); ++i)
  {
    double k = kvalues[i];
    std::string filename = txt_prefix + "perturbations_k" + std::to_string(k*Constants.Mpc) + ".txt";
    pert.output(k, filename);
  }

  //=========================================================================
  // Module IV
  //=========================================================================

  PowerSpectrum power(&cosmo, &rec, &pert, A_s, n_s, kpivot_mpc);
  power.solve();
  power.output_CMB(txt_prefix + "cells.txt");
  power.output_matter(txt_prefix + "matter_power.txt");

  Utils::EndTiming("Everything");
  return 0;

}
