#include "BackgroundCosmology.h"

//====================================================
// Constructors
//====================================================

BackgroundCosmology::BackgroundCosmology(
    double h,
    double OmegaB,
    double OmegaCDM,
    double OmegaK,
    double Neff,
    double TCMB) :
  h(h),
  OmegaB(OmegaB),
  OmegaCDM(OmegaCDM),
  OmegaK(OmegaK),
  Neff(Neff),
  TCMB(TCMB)
{
  // Computing the derived quantities; H0, OmegaR, OmegaNu and OmegaLambda
  H0 = Constants.H0_over_h*h;
  OmegaR = 2*M_PI*M_PI/30.0*std::pow((Constants.k_b*TCMB), 4)/(std::pow(Constants.hbar, 3)*std::pow(Constants.c, 5))
           * 8*M_PI*Constants.G/(3*H0*H0);

  OmegaNu = Neff*7.0/8.0*std::pow((4.0/11.0), 4.0/3.0)*OmegaR;
  OmegaLambda = 1 - (OmegaK + OmegaB + OmegaCDM + OmegaR + OmegaNu);

}

//====================================================
// Do all the solving. Compute eta(x)
//====================================================

// Solve the background
void BackgroundCosmology::solve(){
  Utils::StartTiming("Eta and t");

  //=============================================================================
  // TODO: Set the range of x and the number of points for the splines
  // For this Utils::linspace(x_start, x_end, npts) is useful
  //=============================================================================
  Vector x_array = Utils::linspace(x_start, x_end, 10000);

  // The ODE for deta/dx
  ODEFunction detadx = [&](double x, const double *eta, double *detadx){
    detadx[0] = Constants.c/Hp_of_x(x);

    return GSL_SUCCESS;
  };

  // The ODE for dt/dx
  ODEFunction dtdx = [&](double x, const double *t, double *dtdx){
    dtdx[0] = 1.0/H_of_x(x);

    return GSL_SUCCESS;
  };

  // Initial value of eta
  Vector eta_init{Constants.c/(Hp_of_x(x_start))};

  // Defining ODESolver and solving for eta
  ODESolver ode_eta;

  ode_eta.solve(detadx, x_array, eta_init);

  auto eta_array = ode_eta.get_data_by_component(0);

  eta_of_x_spline.create(x_array, eta_array, "eta_of_x");

  // Defining ODESolver and solving for t
  Vector t_init{1.0/(2.0*H_of_x(x_start))};
  ODESolver ode_t;

  ode_t.solve(dtdx, x_array, t_init);

  auto t_array = ode_t.get_data_by_component(0);

  t_of_x_spline.create(x_array, t_array, "t_of_x");

  Utils::EndTiming("Eta and t");
}

double BackgroundCosmology::comoving_distance(double x) const{
  return eta_of_x_spline(0) - eta_of_x_spline(x);
}

double BackgroundCosmology::r_coordinate(double x) const{
  double chi = comoving_distance(x);

  double argument = std::sqrt(std::abs(OmegaK))*H0*chi/Constants.c;

  if (OmegaK < 0)
  {
    return chi*std::sin(argument)/argument;
  }

  else if (OmegaK > 0)
  {
    return chi*std::sinh(argument)/argument;
  }

  else
  {
    return chi;
  }
}

double BackgroundCosmology::d_A(double x) const{
  double a = std::exp(x);

  return a*r_coordinate(x);
}

double BackgroundCosmology::d_L(double x) const{
  double a = std::exp(x);

  return r_coordinate(x)/a;
}

//====================================================
// Get methods
//====================================================

double BackgroundCosmology::H_of_x(double x) const{

  //=============================================================================
  // TODO: Implement...
  //=============================================================================
  double a = std::exp(x);

  return H0*std::sqrt((OmegaB + OmegaCDM)*std::pow(a, -3) + (OmegaR + OmegaNu)*std::pow(a, -4)
                      + OmegaK*std::pow(a, -2) + OmegaLambda);
}

double BackgroundCosmology::Hp_of_x(double x) const{

  //=============================================================================
  // TODO: Implement...
  //=============================================================================
  double a = std::exp(x);

  return a*H_of_x(x);
}

double BackgroundCosmology::dHdx_of_x(double x) const{
  double a = std::exp(x);

  double Omega_m = OmegaB + OmegaCDM; // Density parameter of non-rel. species
  double Omega_r = OmegaR + OmegaNu; // Density parameter of relativistic species

  // Derivative of H with respect to a
  double dHda = -1.0/2.0*H0*H0/H_of_x(x)*(3*Omega_m*std::pow(a, -4) + 4*Omega_r*std::pow(a, -5));

  // dHdx = da/dx * dH/da = a*dH/da
  return a*dHda;
}

double BackgroundCosmology::ddHddx_of_x(double x) const{
  double a = std::exp(x);

  double Omega_m = OmegaB + OmegaCDM; // Density parameter of non-rel. species
  double Omega_r = OmegaR + OmegaNu; // Density parameter of relativistic species

  double Omega = Omega_m*std::pow(a, -3) + Omega_r*std::pow(a, -4) + OmegaLambda;
  double dOmegada = -3*Omega_m*std::pow(a, -4) - 4*Omega_r*std::pow(a, -5);
  double ddOmegadda = 12*Omega_m*std::pow(a, -5) + 20*Omega_r*std::pow(a, -6);

  double dOmegadx = a*dOmegada;
  double ddOmegaddx = a*dOmegada + a*a*ddOmegadda;

  return H0*(-1.0/(4*std::pow(Omega, 3.0/2.0))*dOmegadx*dOmegadx + 1.0/(2*std::sqrt(Omega))*ddOmegaddx);

}

double BackgroundCosmology::dHpdx_of_x(double x) const{

  //=============================================================================
  // TODO: Implement...
  //=============================================================================
  double a = std::exp(x);
  //double Omega_m = OmegaB + OmegaCDM; // Density parameter of non-rel. species
  //double Omega_r = OmegaR + OmegaNu; // Density parameter of relativistic species

  //return Hp_of_x(x) - 1.0/2.0*H0*H0/H_of_x(x)*(3*Omega_m*std::pow(a, -2) + 4*Omega_r*std::pow(a, -3));
  return Hp_of_x(x) + a*dHdx_of_x(x);
}

double BackgroundCosmology::ddHpddx_of_x(double x) const{

  //=============================================================================
  // TODO: Implement...
  //=============================================================================
  double a = std::exp(x);

  return dHpdx_of_x(x) + a*dHdx_of_x(x) + a*ddHddx_of_x(x);

  /*
  //return dHpdx_of_x(x) + a*dHdx_of_x(x) + a*ddHddx_of_x(x);
  double Omega = Omega_m*std::pow(a, -3) + Omega_r*std::pow(a, -4) + OmegaLambda;
  double dOmegada = -3*Omega_m*std::pow(a, -4) - 4*Omega_r*std::pow(a, -5);
  double ddOmegadda = 12*Omega_m*std::pow(a, -5) + 20*Omega_r*std::pow(a, -6);

  return a*H0*(std::sqrt(Omega) + 3.0/2.0*a/std::sqrt(Omega)*dOmegada \
              - 1.0/4.0*a*a/std::pow(Omega, 3.0/2.0)*dOmegada*dOmegada + a*a/(2.0*std::sqrt(Omega))*ddOmegadda);
  */

}

double BackgroundCosmology::get_OmegaB(double x) const{
  double a = std::exp(x);

  return OmegaB*std::pow(a, -3)*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::get_OmegaR(double x) const{
  double a = std::exp(x);

  return OmegaR*std::pow(a, -4)*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::get_OmegaNu(double x) const{
  double a = std::exp(x);

  return OmegaNu*std::pow(a, -4)*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::get_OmegaCDM(double x) const{
  double a = std::exp(x);

  return OmegaCDM*std::pow(a, -3)*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::get_OmegaLambda(double x) const{
  return OmegaLambda*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::get_OmegaK(double x) const{
  double a = std::exp(a);

  return OmegaK*std::pow(a, -2)*H0*H0/std::pow(H_of_x(x), 2);
}

double BackgroundCosmology::eta_of_x(double x) const{
  return eta_of_x_spline(x);
}

double BackgroundCosmology::t_of_x(double x) const{
  return t_of_x_spline(x);
}

double BackgroundCosmology::get_H0() const{
  return H0;
}

double BackgroundCosmology::get_h() const{
  return h;
}

double BackgroundCosmology::get_Neff() const{
  return Neff;
}

double BackgroundCosmology::get_TCMB(double x) const{
  if(x == 0.0) return TCMB;
  return TCMB * exp(-x);
}

//====================================================
// Print out info about the class
//====================================================
double BackgroundCosmology::age_of_universe(double x) const{
  double age_in_seconds = t_of_x_spline(x);

  return age_in_seconds/(3600.0*24*365*1e9);
}

void BackgroundCosmology::info() const{
  std::cout << "\n";
  std::cout << "Info about cosmology class:\n";
  std::cout << "OmegaB:      " << OmegaB      << "\n";
  std::cout << "OmegaCDM:    " << OmegaCDM    << "\n";
  std::cout << "OmegaLambda: " << OmegaLambda << "\n";
  std::cout << "OmegaK:      " << OmegaK      << "\n";
  std::cout << "OmegaNu:     " << OmegaNu     << "\n";
  std::cout << "OmegaR:      " << OmegaR      << "\n";
  std::cout << "Neff:        " << Neff        << "\n";
  std::cout << "h:           " << h           << "\n";
  std::cout << "TCMB:        " << TCMB        << "\n";
  std::cout << std::endl;
}

void BackgroundCosmology::set_xlims(double x_min, double x_max){
  x_start = x_min;
  x_end = x_max;
}

//====================================================
// Output some data to file
//====================================================
void BackgroundCosmology::output(const std::string filename) const{
  const double x_min = -20.0;
  const double x_max =  5.0;
  const int    n_pts =  10000;

  Vector x_array = Utils::linspace(x_min, x_max, n_pts);

  std::ofstream fp(filename.c_str());
  auto print_data = [&] (const double x) {
    fp << x                  << " ";
    fp << eta_of_x(x)        << " ";
    fp << Hp_of_x(x)         << " ";
    fp << t_of_x(x)          << " ";
    fp << dHpdx_of_x(x)      << " ";
    fp << ddHpddx_of_x(x)    << " ";
    fp << get_OmegaB(x)      << " ";
    fp << get_OmegaCDM(x)    << " ";
    fp << get_OmegaLambda(x) << " ";
    fp << get_OmegaR(x)      << " ";
    fp << get_OmegaNu(x)     << " ";
    fp << get_OmegaK(x)      << " ";
    fp <<"\n";
  };
  std::for_each(x_array.begin(), x_array.end(), print_data);
}
