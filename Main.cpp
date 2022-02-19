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
  cosmo.set_xlims(-20, 5); // Setting the x-range
  cosmo.solve();
  cosmo.info();

  double age = cosmo.age_of_universe(0);

  std::cout << "Age of the universe: " << age << " Gyr\n";

  // Defining x-array that is used when computing luminosity distances
  // that are compared with observed lum. distances
  Vector x_array = Utils::linspace(-1, 0, 100);
  std::ofstream d_L{"lum_dist.txt"};

  // Writing x and lum. distance to file for each value of x
  for (int i = 0; i < x_array.size(); ++i)
  {
    d_L << x_array[i] << "\t" << cosmo.d_L(x_array[i]) << "\n";
  }


  // Output background evolution quantities
  cosmo.output("cosmology.txt");

  Utils::EndTiming("Everything");

  return 0;
}
