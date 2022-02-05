#include "Utils.h"
#include "BackgroundCosmology.h"

int main(int argc, char **argv){
  Utils::StartTiming("Everything");

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

  //=========================================================================
  // Module I
  //=========================================================================

  // Set up and solve the background
  BackgroundCosmology cosmo(h, OmegaB, OmegaCDM, OmegaK, Neff, TCMB);
  cosmo.set_xlims(-20, 5);
  cosmo.solve();
  cosmo.info();

  double x = std::log(1e-3);
  double age = cosmo.age_of_universe(0);

  std::cout << "Age of the universe: " << age << " Gyr\n";

  std::cout << "d_A = " << cosmo.d_L(x)/(3.08567758e22) << "\n";

  // Output background evolution quantities
  cosmo.output("cosmology.txt");

  Utils::EndTiming("Everything");

  return 0;
}
