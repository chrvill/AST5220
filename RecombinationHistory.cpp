#include"RecombinationHistory.h"

//====================================================
// Constructors
//====================================================

RecombinationHistory::RecombinationHistory(
    BackgroundCosmology *cosmo,
    double Yp) :
  cosmo(cosmo),
  Yp(Yp)
{}

//====================================================
// Do all the solving we need to do
//====================================================

void RecombinationHistory::solve(){

  // Compute and spline Xe, ne
  solve_number_density_electrons();

  // Compute and spline tau, dtaudx, ddtauddx, g, dgdx, ddgddx, ...
  solve_for_optical_depth_tau();
}

//====================================================
// Solve for X_e and n_e using Saha and Peebles and spline the result
//====================================================

void RecombinationHistory::solve_number_density_electrons(){
  Utils::StartTiming("Xe");

  //=============================================================================
  // TODO: Set up x-array and make arrays to store X_e(x) and n_e(x) on
  //=============================================================================

  Vector x_array = Utils::linspace(x_start, x_end, npts_rec_arrays);
  Vector Xe_arr = Utils::linspace(x_start, x_end, npts_rec_arrays);
  Vector ne_arr = Utils::linspace(x_start, x_end, npts_rec_arrays);

  // Calculate recombination history
  bool saha_regime = true;
  for(int i = 0; i < npts_rec_arrays; i++){

    //==============================================================
    // TODO: Get X_e from solving the Saha equation so
    // implement the function electron_fraction_from_saha_equation
    //==============================================================
    auto Xe_ne_data = electron_fraction_from_saha_equation_with_He(x_array[i]);

    // Electron fraction and number density
    const double Xe_current = Xe_ne_data.first;
    const double ne_current = Xe_ne_data.second;

    // Are we still in the Saha regime?
    if(Xe_current < Xe_saha_limit)
      saha_regime = false;

    if(saha_regime){

      //=============================================================================
      // TODO: Store the result we got from the Saha equation
      //=============================================================================
      Xe_arr[i] = Xe_current;
      ne_arr[i] = ne_current;

    } else {

      //==============================================================
      // TODO: Compute X_e from current time til today by solving
      // the Peebles equation (NB: if you solve all in one go remember to
      // exit the for-loop!)
      // Implement rhs_peebles_ode
      //==============================================================

      // The Peebles ODE equation
      ODESolver peebles_Xe_ode;
      ODEFunction dXedx = [&](double x, const double *Xe, double *dXedx){
        return rhs_peebles_ode(x, Xe, dXedx);
      };

      //=============================================================================
      // TODO: Set up IC, solve the ODE and fetch the result
      //=============================================================================

      Vector Xe_init{Xe_arr[i - 1]};

      Vector x_arr_peebles(npts_rec_arrays - i);

      for (int j = i; j < npts_rec_arrays; ++j)
      {
        x_arr_peebles[j - i] = x_array[j];
      }

      peebles_Xe_ode.solve(dXedx, x_arr_peebles, Xe_init);
      auto temp_Xe_array = peebles_Xe_ode.get_data_by_component(0);

      //Xe_arr.insert(Xe_arr.begin() + i, temp_Xe_array.begin(), temp_Xe_array.end());

      for (int j = 0; j < x_arr_peebles.size(); ++j)
      {
        Xe_arr[j + i] = temp_Xe_array[j];
      }

      /*
      for (int j = 0; j < npts_rec_arrays; ++j)
      {
        std::cout << x_array[j] << "\t" << Xe_arr[j] << "\n";
      }
      */

      break;
    }
  }

  //=============================================================================
  // TODO: Spline the result. Implement and make sure the Xe_of_x, ne_of_x
  // functions are working
  //=============================================================================

  std::for_each(Xe_arr.begin(), Xe_arr.end(), [&](double& x){x = std::log(x);});

  log_Xe_of_x_spline.create(x_array, Xe_arr, "Xe");

  Utils::EndTiming("Xe");
}

//====================================================
// Solve the Saha equation to get ne and Xe
//====================================================
std::pair<double,double> RecombinationHistory::electron_fraction_from_saha_equation(double x) const{
  const double a           = exp(x);

  // Physical constants
  const double k_b         = Constants.k_b;
  const double G           = Constants.G;
  const double m_e         = Constants.m_e;
  const double hbar        = Constants.hbar;
  const double m_H         = Constants.m_H;
  const double epsilon_0   = Constants.epsilon_0;
  const double H0_over_h   = Constants.H0_over_h;

  //const double H0 = H0_over_h*cosmo->get_h();
  //const double rho_crit_0 = 3*H0*H0/(8*M_PI*G);

  // Fetch cosmological parameters
  //const double OmegaB = cosmo->get_OmegaB(0);
  const double T_b  = cosmo->get_TCMB(x);
  const double n_b  = cosmo->get_n_b(x);

  // Electron fraction and number density
  double Xe = 0.0;
  double ne = 0.0;

  //=============================================================================
  // TODO: Compute Xe and ne from the Saha equation
  //=============================================================================
  const double argument = m_e*k_b*T_b/(2.0*M_PI*hbar*hbar);
  const double C = 1.0/n_b*std::pow(argument, 3.0/2.0)*std::exp(-epsilon_0/(k_b*T_b));

  // A simplification of the citardauq formula, where I've assumed sqrt(1 + 4/C) ~ 1 + 2/C
  // this is always smaller than or equal to one (which is what we want), so is
  // much more stable than the standard quadratic formula, which jumps around 1 (both above and below)
  Xe = 1.0/(1.0 + 1.0/C);
  ne = Xe*n_b;

  return std::pair<double,double>(Xe, ne);
}

std::pair<double,double> RecombinationHistory::electron_fraction_from_saha_equation_with_He(double x) const{
  const double a           = exp(x);

  // Physical constants
  const double k_b         = Constants.k_b;
  const double G           = Constants.G;
  const double m_e         = Constants.m_e;
  const double hbar        = Constants.hbar;
  const double m_H         = Constants.m_H;
  const double epsilon_0   = Constants.epsilon_0;
  const double H0_over_h   = Constants.H0_over_h;
  const double xhi0        = Constants.xhi0;
  const double xhi1        = Constants.xhi1;

  // Fetch cosmological parameters
  //const double OmegaB = cosmo->get_OmegaB(0);
  const double T_b  = cosmo->get_TCMB(x);
  const double n_b  = cosmo->get_n_b(x);

  // Electron fraction and number density
  double Xe = 0.0;
  double ne = 0.0;

  const double parenthesis_factor = std::pow(m_e*k_b*T_b/(2*M_PI*hbar*hbar), 3.0/2.0);

  const double C_1 = 2*parenthesis_factor*std::exp(-xhi0/(k_b*T_b));
  const double C_2 = 4*parenthesis_factor*std::exp(-xhi1/(k_b*T_b));
  const double C_3 = parenthesis_factor*std::exp(-epsilon_0/(k_b*T_b));

  // Initial guess for f_e for every value of x when using the Saha eq.
  double f_e = 0.999;
  // Initializing variable that will contain the old guess for f_e
  double f_e_old = 0;

  while (std::abs(f_e - f_e_old) > 1e-10)
  {
    ne = f_e*n_b;

    double x_H_plus = C_3/(ne + C_3);
    double x_He_plus = C_1/(ne + C_1 + C_1*C_2/ne);
    double x_He_pplus = C_2/ne*x_He_plus;

    f_e = (2*x_He_pplus + x_He_plus)*Yp/4.0 + x_H_plus*(1 - Yp);
    f_e_old = f_e;
  }

  Xe = f_e/(1 - Yp);

  return std::pair<double,double>(Xe, ne);
}

//====================================================
// The right hand side of the dXedx Peebles ODE
//====================================================
int RecombinationHistory::rhs_peebles_ode(double x, const double *Xe, double *dXedx){

  // Current value of a and X_e
  const double X_e         = Xe[0];
  const double a           = exp(x);

  // Physical constants in SI units
  const double k_b         = Constants.k_b;
  const double G           = Constants.G;
  const double c           = Constants.c;
  const double m_e         = Constants.m_e;
  const double hbar        = Constants.hbar;
  const double m_H         = Constants.m_H;
  const double sigma_T     = Constants.sigma_T;
  const double lambda_2s1s = Constants.lambda_2s1s;
  const double epsilon_0   = Constants.epsilon_0;

  // Cosmological parameters
  const double H      = cosmo->H_of_x(x);
  const double T_b    = cosmo->get_TCMB(x);
  const double n_b    = cosmo->get_n_b(x);

  //=============================================================================
  // TODO: Write the expression for dXedx
  //=============================================================================

  const double phi_2        = 0.448*std::log(epsilon_0/(k_b*T_b));
  const double alpha_2      = 8*sigma_T*c/std::sqrt(3*M_PI)*std::sqrt(epsilon_0/(k_b*T_b))*phi_2;
  const double beta         = alpha_2*std::pow((m_e*k_b*T_b/(2.0*M_PI*hbar*hbar)), 3.0/2.0) \
                              *std::exp(-epsilon_0/(k_b*T_b));

  // Have written beta_2 explicitly, instead of in terms of beta. The former is more stable for low temperatures.
  const double beta_2       = alpha_2*std::pow((m_e*k_b*T_b/(2.0*M_PI*hbar*hbar)), 3.0/2.0) \
                              *std::exp(-epsilon_0/(4*k_b*T_b));

  const double n_H          = n_b*(1 - Yp);
  const double n_1s         = (1 - X_e)*n_H;
  const double Lambda_alpha = H*std::pow((3*epsilon_0/(hbar*c)), 3)/(8*8*M_PI*M_PI*n_1s);
  const double C_r          = (lambda_2s1s + Lambda_alpha)/(lambda_2s1s + Lambda_alpha + beta_2);

  dXedx[0] = C_r/H*(beta*(1.0 - X_e) - n_H*alpha_2*X_e*X_e);

  return GSL_SUCCESS;
}

//====================================================
// Solve for the optical depth tau, compute the
// visibility function and spline the result
//====================================================

void RecombinationHistory::solve_for_optical_depth_tau(){
  Utils::StartTiming("opticaldepth");

  // Set up x-arrays to integrate over. We split into three regions as we need extra points in reionisation
  const int npts = 1000;
  Vector x_array = Utils::linspace(x_start, x_end, npts);

  // Reversing x_array because we want to integrate from today and back in time
  std::reverse(x_array.begin(), x_array.end());

  // The ODE system dtau/dx, dtau_noreion/dx and dtau_baryon/dx
  ODEFunction dtaudx = [&](double x, const double *tau, double *dtaudx){

    //=============================================================================
    // TODO: Write the expression for dtaudx
    //=============================================================================

    // Set the derivative for photon optical depth
    const double ne = ne_of_x(x);
    const double H  = cosmo->H_of_x(x);

    dtaudx[0] = -Constants.c*ne*Constants.sigma_T/H;

    return GSL_SUCCESS;
  };

  //=============================================================================
  // TODO: Set up and solve the ODE and make tau splines
  //=============================================================================
  Vector tau_init{0};

  ODESolver tau_ode;
  tau_ode.solve(dtaudx, x_array, tau_init);
  auto tau_array = tau_ode.get_data_by_component(0);

  std::reverse(x_array.begin(), x_array.end());
  std::reverse(tau_array.begin(), tau_array.end());

  /*
  for (int i = 0; i < npts; ++i)
  {
    std::cout << tau_array[i] << "\t" << x_array[i] << "\n";
  }
  */


  tau_of_x_spline.create(x_array, tau_array, "tau");

  //=============================================================================
  // TODO: Compute visibility functions and spline everything
  //=============================================================================
  Vector g_tilde(npts);
  for (int i = 0; i < npts; i++)
  {
    //std::cout << std::exp(-tau_array[i]) << "\n";
    double g_tilde_current = -dtaudx_of_x(x_array[i])*std::exp(-tau_array[i]);
    g_tilde[i] = g_tilde_current;
  }

  g_tilde_of_x_spline.create(x_array, g_tilde, "g");

  Utils::EndTiming("opticaldepth");
}

//====================================================
// Get methods
//====================================================

double RecombinationHistory::tau_of_x(double x) const{
  return tau_of_x_spline(x);
}

double RecombinationHistory::dtaudx_of_x(double x) const{

  //=============================================================================
  // TODO: Implement. Either from the tau-spline tau_of_x_spline.deriv_(x) or
  // from a separate spline if you choose to do this
  //=============================================================================

  return tau_of_x_spline.deriv_x(x);
}

double RecombinationHistory::ddtauddx_of_x(double x) const{

  //=============================================================================
  // TODO: Implement
  //=============================================================================
  return tau_of_x_spline.deriv_xx(x);
}

double RecombinationHistory::g_tilde_of_x(double x) const{
  return g_tilde_of_x_spline(x);
}

double RecombinationHistory::dgdx_tilde_of_x(double x) const{

  //=============================================================================
  // TODO: Implement
  //=============================================================================

  return g_tilde_of_x_spline.deriv_x(x);
}

double RecombinationHistory::ddgddx_tilde_of_x(double x) const{

  //=============================================================================
  // TODO: Implement
  //=============================================================================
  return g_tilde_of_x_spline.deriv_xx(x);
}

double RecombinationHistory::Xe_of_x(double x) const{

  //=============================================================================
  // TODO: Implement
  //=============================================================================
  return std::exp(log_Xe_of_x_spline(x));

}

double RecombinationHistory::ne_of_x(double x) const{

  //=============================================================================
  // TODO: Implement
  //=============================================================================
  double Xe = Xe_of_x(x);
  double n_H = (1 - Yp)*cosmo->get_n_b(x);

  return Xe*n_H;
}

double RecombinationHistory::get_Yp() const{
  return Yp;
}

//====================================================
// Print some useful info about the class
//====================================================
void RecombinationHistory::info() const{
  std::cout << "\n";
  std::cout << "Info about recombination/reionization history class:\n";
  std::cout << "Yp:          " << Yp          << "\n";
  std::cout << std::endl;
}

//====================================================
// Output the data computed to file
//====================================================
void RecombinationHistory::output(const std::string filename) const{
  std::ofstream fp(filename.c_str());
  const int npts       = 5000;
  const double x_min   = x_start;
  const double x_max   = x_end;

  Vector x_array = Utils::linspace(x_min, x_max, npts);
  auto print_data = [&] (const double x) {
    fp << x                    << " ";
    fp << Xe_of_x(x)           << " ";
    fp << ne_of_x(x)           << " ";
    fp << tau_of_x(x)          << " ";
    fp << dtaudx_of_x(x)       << " ";
    fp << ddtauddx_of_x(x)     << " ";
    fp << g_tilde_of_x(x)      << " ";
    fp << dgdx_tilde_of_x(x)   << " ";
    fp << ddgddx_tilde_of_x(x) << " ";
    fp << "\n";
  };
  std::for_each(x_array.begin(), x_array.end(), print_data);
}
