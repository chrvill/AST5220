#include"PowerSpectrum.h"

//====================================================
// Constructors
//====================================================

PowerSpectrum::PowerSpectrum(
    BackgroundCosmology *cosmo,
    RecombinationHistory *rec,
    Perturbations *pert,
    double A_s,
    double n_s,
    double kpivot_mpc) :
  cosmo(cosmo),
  rec(rec),
  pert(pert),
  A_s(A_s),
  n_s(n_s),
  kpivot_mpc(kpivot_mpc)
{}

//====================================================
// Do all the solving
//====================================================
void PowerSpectrum::solve(){

  Vector k_array = Utils::linspace(k_min, k_max, 2000);

  generate_bessel_function_splines();

  line_of_sight_integration(k_array);

  // Number of points per oscillation in the integrand of C_ell
  const int n = 32;
  const double eta_0 = cosmo->eta_of_x(0);
  const double delta_k = 2*M_PI/(n*eta_0);

  // Total numebr of points at which we evaluate the integrand of C_ell
  const int n_k = std::floor((k_max - k_min)/delta_k);

  k_array = Utils::linspace(k_min, k_max, n_k);

  // Integrating C_ell and splining the result
  auto cell_TT = solve_for_cell(k_array, thetaT_ell_of_k_spline, thetaT_ell_of_k_spline);
  cell_TT_spline.create(ells, cell_TT, "Cell_TT_of_ell");
}

//====================================================
// Generate splines of j_ell(z) needed for LOS integration
//====================================================

void PowerSpectrum::generate_bessel_function_splines(){
  Utils::StartTiming("besselspline");

  // Make storage for the splines
  j_ell_splines = std::vector<Spline>(ells.size());

  const double eta_0 = cosmo->eta_of_x(0);
  const double n = 16; // Number of points per oscillation of j_ell

  const double delta_x = 2*M_PI/n;
  const double x_min = 0;
  const double x_max = k_max*eta_0;

  // Number of points at which we evaluate j_ell
  const int n_points = std::ceil((x_max - x_min)/delta_x);

  Vector x = Utils::linspace(x_min, x_max, n_points);

  for(size_t i = 0; i < ells.size(); i++){
    const int ell = ells[i];

    Vector j_ell_array(n_points);

    // Evaluating j_ell for each value of x
    for (int ix = 0; ix < n_points; ++ix)
    {
      j_ell_array[ix] = Utils::j_ell(ell, x[ix]);
    }

    // Make the j_ell_splines[i] spline
    j_ell_splines[i].create(x, j_ell_array);

  }

  Utils::EndTiming("besselspline");
}

//====================================================
// Do the line of sight integration for a single
// source function
//====================================================

Vector2D PowerSpectrum::line_of_sight_integration_single(
    Vector & k_array,
    std::function<double(double,double)> &source_function){
  Utils::StartTiming("lineofsight");

  // Make storage for the results
  Vector2D result = Vector2D(ells.size(), Vector(k_array.size()));

  const double eta_0 = cosmo->eta_of_x(0);

  // Number of points at which we evaluate the integrand in the line of sight integral
  const int n_points_x = 1000;
  const double delta_x = (Constants.x_end - Constants.x_start)/n_points_x;
  Vector x_array = Utils::linspace(Constants.x_start, Constants.x_end, n_points_x);

  // Computing Theta_ell for each l in ells, and for every k in k_array
  #pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < ells.size(); ++i)
  {

    for (int ik = 0; ik < k_array.size(); ++ik)
    {
      double integral = 0;
      const double k = k_array[ik];

      // Sampling the integrand of Theta_ell at n_points_x points. Using the trapezoidal rule
      // to compute the integral
      for (int ix = 0; ix < n_points_x; ++ix)
      {
        const double x   = x_array[ix];
        const double eta = cosmo->eta_of_x(x);

        // The trapezoidal rule for integration:
        // int_a^b f(x) dx = [sum(x from a to b)(f(a) + 2*f(a + delta_x) + ... + 2*f(b - delta_x) + f(b))]*delta_x/2
        // The multiplication by delta_x/2 is done at the end, after the loop over x, since the x-points
        // are linearly spaced.
        if (ix == 0 || ix == (n_points_x - 1))
        {
          integral += 1.0*source_function(x, k)*j_ell_splines[i](k*(eta_0 - eta));
        }
        else
        {
          integral += 2.0*source_function(x, k)*j_ell_splines[i](k*(eta_0 - eta));
        }
      }

      integral *= delta_x/2.0;
      result[i][ik] = integral;
      }
    }

  Utils::EndTiming("lineofsight");
  return result;
}

//====================================================
// Do the line of sight integration
//====================================================
void PowerSpectrum::line_of_sight_integration(Vector & k_array){
  const int n_k        = k_array.size();
  const int nells      = ells.size();

  // Make storage for the splines we are to create
  thetaT_ell_of_k_spline = std::vector<Spline>(nells);

  // Make a function returning the source function
  std::function<double(double,double)> source_function_T = [&](double x, double k){
    return pert->get_Source_T(x,k);
  };

  // Do the line of sight integration
  Vector2D thetaT_ell_of_k = line_of_sight_integration_single(k_array, source_function_T);

  // Splining Theta_ell
  for (int i = 0; i < nells; ++i)
  {
    thetaT_ell_of_k_spline[i].create(k_array, thetaT_ell_of_k[i]);
  }
}

//====================================================
// Compute Cell (here only TT)
// Cell = Int_0^inf 4 * pi * P(k) f_ell g_ell dk/k
//====================================================
Vector PowerSpectrum::solve_for_cell(
    Vector &k_array,
    std::vector<Spline> &f_ell_spline,
    std::vector<Spline> &g_ell_spline){
  const int nells      = ells.size();

  Vector result(nells);

  // The integrand in the expression for C_ell
  auto dCell_dk = [&](const double ell_index, const double k)
  {
    return 4*M_PI*primordial_power_spectrum(k)*f_ell_spline[ell_index](k)*g_ell_spline[ell_index](k)/k;
  };

  //Vector ells_print = {10, 100, 1000};
  // Indices corresponding to the values of ell for which we want to write
  // Theta_ell to file. Corresponds to ell = 10, 100 and 1000
  Vector ells_indices = {7, 19, 42};

  Vector k_print = Utils::linspace(k_min, k_max, 10000);

  // Writing Theta_ell to file for these values of ell
  for (int i = 0; i < ells_indices.size(); ++i)
  {
    const double ell = ells[ells_indices[i]];

    std::string filename = "txtfiles/theta" + std::to_string(static_cast<int>(ell)) + ".txt";
    std::ofstream fp(filename);

    for (int ik = 0; ik < k_print.size(); ++ik)
    {
      const double Theta_ell = f_ell_spline[ells_indices[i]](k_print[ik]);

      fp << k_print[ik] << "\t" << Theta_ell << "\n";
    }
  }
  // delta_k is needed in the trapezoidal rule, when multiplying by delta_k/2
  const double delta_k = (k_array[k_array.size() - 1] - k_array[0])/k_array.size();

  // Again using trapezoidal rule to compute the integral, this time an integral over k
  #pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < nells; ++i)
  {
    double integral = 0;
    const double ell = ells[i];

    for (int ik = 0; ik < k_array.size(); ++ik)
    {
      const double k = k_array[ik];

      if (ik == 0 || ik == (k_array.size() - 1))
      {
        integral += 1.0*dCell_dk(i, k);
      }
      else
      {
        integral += 2.0*dCell_dk(i, k);
      }
    }

    integral *= delta_k/2.0;
    result[i] = integral;
  }

  return result;
}

//====================================================
// The primordial power-spectrum
//====================================================

double PowerSpectrum::primordial_power_spectrum(const double k) const{
  return A_s*pow(Constants.Mpc*k/kpivot_mpc, n_s - 1.0);
}

double PowerSpectrum::get_matter_power_spectrum(const double x, const double k) const{
  // Background cosmological and perturbation variables needed to compute the matter power spectrum
  const double H0       = cosmo->H_of_x(0);
  const double Omega_M0 = cosmo->get_OmegaM(0);
  const double a        = exp(x);
  const double c        = Constants.c;
  const double Psi = pert->get_Psi(x, k);

  const double Delta_M = pow(c*k/H0, 2)*Psi*2.0/3.0*a/Omega_M0;

  return abs(Delta_M)*abs(Delta_M)*primordial_power_spectrum(k)*2*M_PI*M_PI/(k*k*k);
}

//====================================================
// Get methods
//====================================================
double PowerSpectrum::get_cell_TT(const double ell) const{
  return cell_TT_spline(ell);
}

//====================================================
// Output the cells to file
//====================================================

void PowerSpectrum::output_CMB(std::string filename) const{
  // Output in standard units of muK^2
  std::ofstream fp(filename.c_str());
  const int ellmax = int(ells[ells.size()-1]);
  auto ellvalues = Utils::linspace(2, ellmax, ellmax-1);

  auto print_data = [&] (const double ell) {
    double normfactor  = (ell * (ell+1)) / (2.0 * M_PI) * pow(1e6 * cosmo->get_TCMB(), 2);
    double normfactorN = (ell * (ell+1)) / (2.0 * M_PI)
      * pow(1e6 * cosmo->get_TCMB() *  pow(4.0/11.0, 1.0/3.0), 2);
    double normfactorL = (ell * (ell+1)) * (ell * (ell+1)) / (2.0 * M_PI);
    fp << ell                                 << " ";
    fp << cell_TT_spline( ell ) * normfactor  << " ";
    fp << "\n";
  };
  std::for_each(ellvalues.begin(), ellvalues.end(), print_data);
}
void PowerSpectrum::output_matter(std::string filename) const{
  // Outputting the matter power spectrum
  std::ofstream fp(filename.c_str());

  const double nk = 1000;

  Vector exponents = Utils::linspace(log(k_min), log(k_max), nk);

  Vector k_array(nk);

  for (int i = 0; i < nk; ++i)
  {
    k_array[i] = exp(exponents[i]);
  }

  auto print_data = [&](const double k){
    fp << k << " ";
    fp << get_matter_power_spectrum(0, k) << " ";
    fp << "\n";
  };

  std::for_each(k_array.begin(), k_array.end(), print_data);
}
