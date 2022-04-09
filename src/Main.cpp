#include "Utils.h"
#include "BackgroundCosmology.h"
#include "RecombinationHistory.h"
#include "Perturbations.h"

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
  //pert.solve();
  pert.info();

  // Output perturbation quantities
  Vector kvalues{0.1/Constants.Mpc, 0.01/Constants.Mpc, 0.001/Constants.Mpc,
                 0.1*h/Constants.Mpc, 0.01*h/Constants.Mpc, 0.001*h/Constants.Mpc};

  //pert.integrate_perturbations();
  pert.solve();

  for (int i = 0; i < kvalues.size(); ++i)
  {
    double k = kvalues[i];
    std::string filename = txt_prefix + "perturbations_k" + std::to_string(k*Constants.Mpc) + ".txt";
    pert.output(k, filename);
  }


  Utils::EndTiming("Everything");
  // Remove when module is completed
  return 0;

}
