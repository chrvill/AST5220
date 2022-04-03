#include"Perturbations.h"

//====================================================
// Constructors
//====================================================

Perturbations::Perturbations(
    BackgroundCosmology *cosmo,
    RecombinationHistory *rec) :
  cosmo(cosmo),
  rec(rec)
{
  Theta_spline = std::vector<Spline2D>(Constants.n_ell_theta);
}

//====================================================
// Do all the solving
//====================================================

void Perturbations::solve(){

  // Integrate all the perturbation equation and spline the result
  integrate_perturbations();

  // Compute source functions and spline the result
  compute_source_functions();
}

//====================================================
// The main work: integrate all the perturbations
// and spline the results
//====================================================

void Perturbations::integrate_perturbations(){
  Utils::StartTiming("integrateperturbation");

  Vector x_array = Utils::linspace(x_start, x_end, n_x);
  //Vector y = Vector(n_x*n_k);

  //===================================================================
  // TODO: Set up the k-array for the k's we are going to integrate over
  // Start at k_min end at k_max with n_k points with either a
  // quadratic or a logarithmic spacing
  //===================================================================
  //Vector k_array = Utils::linspace(k_min, k_max, n_k);

  Vector exponents = Utils::linspace(log(k_min), log(k_max), n_k);

  Vector k_array(n_k);

  for (int i = 0; i < n_k; ++i)
  {
    k_array[i] = exp(exponents[i]);
  }

  //Vector k_array{0.1/Constants.Mpc};

  Vector delta_cdm(n_x*n_k);
  Vector delta_b(n_x*n_k);
  Vector v_cdm(n_x*n_k);
  Vector v_b(n_x*n_k);
  Vector Phi(n_x*n_k);

  Vector2D Thetas(Constants.n_ell_theta, Vector(n_x*n_k, 0.0));

  #pragma omp parallel for schedule(dynamic, 8)
  // Loop over all wavenumbers
  for(int ik = 0; ik < n_k; ik++){

    /*
    // Progress bar...
    if( (10*ik) / n_k != (10*ik+10) / n_k ) {
      std::cout << (100*ik+100)/n_k << "% " << std::flush;
      if(ik == n_k-1) std::cout << std::endl;
    }
    */

    Vector2D y(n_x, Vector(Constants.n_ell_tot_full, 0.0));

    // Current value of k
    double k = k_array[ik];
    std::cout << k << "\n";

    // Find value to integrate to
    double x_end_tight = get_tight_coupling_time(k);
    //std::cout << x_end_tight << "\n\n";

    //===================================================================
    // TODO: Tight coupling integration
    // Remember to implement the routines:
    // set_ic : The IC at the start
    // rhs_tight_coupling_ode : The dydx for our coupled ODE system
    //===================================================================

    // Set up initial conditions in the tight coupling regime
    auto y_tight_coupling_ini = set_ic(x_start, k);

    // The tight coupling ODE system
    ODEFunction dydx_tight_coupling = [&](double x, const double *y, double *dydx){
      return rhs_tight_coupling_ode(x, k, y, dydx);
    };

    const int index_end_tight = std::lower_bound(x_array.begin(), x_array.end(), x_end_tight) - x_array.begin();

    Vector x_tight;
    x_tight = Vector(x_array.begin(), x_array.begin() + index_end_tight);

    ODESolver tc_ode;
    tc_ode.solve(dydx_tight_coupling, x_tight, y_tight_coupling_ini);
    auto y_tight_coupling = tc_ode.get_data();

    Vector y_end_tight = y_tight_coupling.back();

    /*
    std::cout << "\n";
    for (int ix = 0; ix < index_end_tight - 1; ++ix)
    {
      std::cout << y_tight_coupling[ix][4]<< "\t" << x_array[ix] << "\n";
    }
    */

    //====================================================================
    // TODO: Full equation integration
    // Remember to implement the routines:
    // set_ic_after_tight_coupling : The IC after tight coupling ends
    // rhs_full_ode : The dydx for our coupled ODE system
    //===================================================================

    Vector x_after_tight;
    // The minus 1 is because the initial conditions for the full system are the last values in y_tight_coupling,
    // which corresponds to x = x_array.begin() + index_end_tight - 1.
    x_after_tight = Vector(x_array.begin() + index_end_tight - 1, x_array.end());

    // Set up initial conditions
    auto y_full_ini = set_ic_after_tight_coupling(y_end_tight, x_after_tight[0], k);

    // The full ODE system
    ODEFunction dydx_full = [&](double x, const double *y, double *dydx){
      return rhs_full_ode(x, k, y, dydx);
    };

    // Integrate from x_end_tight -> x_end
    ODESolver full_ode;
    full_ode.solve(dydx_full, x_after_tight, y_full_ini);
    auto y_full = full_ode.get_data();

    // Inserting the contents of y_tight_coupling into y.
    // This was the best way I found to insert the tight-coupling
    // variables while still keeping the zeros corresponding to the higher order
    // multipoles of the photons
    for (int ix = 0; ix < index_end_tight - 1; ++ix)
    {
      std::copy(y_tight_coupling[ix].begin(), y_tight_coupling[ix].end(), y[ix].begin());
    }

    // Inserting the contents of y_full into y
    std::copy(y_full.begin(), y_full.end(), y.begin() + index_end_tight - 1);

    for (int ix = 0; ix < n_x; ++ix)
    {
      int index = ix + n_x*ik;

      delta_cdm[index] = y[ix][Constants.ind_deltacdm];
      delta_b[index]   = y[ix][Constants.ind_deltab];
      v_cdm[index]     = y[ix][Constants.ind_vcdm];
      v_b[index]       = y[ix][Constants.ind_vb];
      Phi[index]       = y[ix][Constants.ind_Phi];

      for (int l = 0; l < Constants.n_ell_theta; ++l)
      {
        Thetas[l][index] = y[ix][Constants.ind_start_theta + l];
      }
    }

    //===================================================================
    // TODO: remember to store the data found from integrating so we can
    // spline it below
    //
    // To compute a 2D spline of a function f(x,k) the data must be given
    // to the spline routine as a 1D array f_array with the points f(ix, ik)
    // stored as f_array[ix + n_x * ik]
    // Example:
    // Vector x_array(n_x);
    // Vector k_array(n_k);
    // Vector f(n_x * n_k);
    // Spline2D y_spline;
    // f_spline.create(x_array, k_array, f_array);
    // We can now use the spline as f_spline(x, k)
    //
    // NB: If you use Theta_spline then you have to allocate it first,
    // before using it e.g.
    // Theta_spline = std::vector<Spline2D>(n_ell_theta);
    //
    //===================================================================


  }
  Utils::EndTiming("integrateperturbation");

  //=============================================================================
  // TODO: Make all splines needed: Theta0,Theta1,Theta2,Phi,Psi,...
  //=============================================================================

  delta_cdm_spline.create(x_array, k_array, delta_cdm);
  delta_b_spline.create(x_array, k_array, delta_b);
  v_cdm_spline.create(x_array, k_array, v_cdm);
  v_b_spline.create(x_array, k_array, v_b);
  Phi_spline.create(x_array, k_array, Phi);

  for (int l = 0; l < Constants.n_ell_theta; ++l)
  {
    std::string spline_name = "theta_" + std::to_string(l) + "_spline";

    Theta_spline[l].create(x_array, k_array, Thetas[l], spline_name);
  }

}

//====================================================
// Set IC at the start of the run (this is in the
// tight coupling regime)
//====================================================
Vector Perturbations::set_ic(const double x, const double k) const{

  const double c = Constants.c;
  const double Hp = cosmo->Hp_of_x(x);
  const double OmegaNu0 = cosmo->get_OmegaNu(0);

  const double ck_Hp = c*k/Hp;

  // The vector we are going to fill
  Vector y_tc(Constants.n_ell_tot_tc);

  //=============================================================================
  // Compute where in the y_tc array each component belongs
  // This is just an example of how to do it to make it easier
  // Feel free to organize the component any way you like
  //=============================================================================

  // For integration of perturbations in tight coupling regime (Only 2 photon multipoles + neutrinos needed)
  const int n_ell_theta_tc      = Constants.n_ell_theta_tc;
  const int n_ell_neutrinos_tc  = Constants.n_ell_neutrinos_tc;
  const int n_ell_tot_tc        = Constants.n_ell_tot_tc;
  const bool polarization       = Constants.polarization;
  const bool neutrinos          = Constants.neutrinos;

  // References to the tight coupling quantities
  double &delta_cdm    =  y_tc[Constants.ind_deltacdm_tc];
  double &delta_b      =  y_tc[Constants.ind_deltab_tc];
  double &v_cdm        =  y_tc[Constants.ind_vcdm_tc];
  double &v_b          =  y_tc[Constants.ind_vb_tc];
  double &Phi          =  y_tc[Constants.ind_Phi_tc];
  double *Theta        = &y_tc[Constants.ind_start_theta_tc];
  //double *Nu           = &y_tc[Constants.ind_start_nu_tc];

  //=============================================================================
  // TODO: Set the initial conditions in the tight coupling regime
  //=============================================================================

  // SET: Scalar quantities (Gravitational potential, baryons and CDM)
  double Psi = -1.0/(3.0/2.0);
  Phi = -Psi;

  delta_cdm = -3.0/2.0*Psi;
  delta_b = delta_cdm;

  v_cdm = -c*k/(2.0*Hp)*Psi;
  v_b = v_cdm;

  // SET: Photon temperature perturbations (Theta_ell)
  double *Theta_0 = Theta;
  double *Theta_1 = Theta + 1;

  *Theta_0 = -1.0/2.0*Psi;
  *Theta_1 = ck_Hp/6.0*Psi;

  // SET: Neutrino perturbations (N_ell)
  if(neutrinos){
  }

  return y_tc;
}

//====================================================
// Set IC for the full ODE system after tight coupling
// regime ends
//====================================================

Vector Perturbations::set_ic_after_tight_coupling(
    const Vector &y_tc,
    const double x,
    const double k) const{

  const double Hp     = cosmo->Hp_of_x(x);
  const double dtaudx = rec->dtaudx_of_x(x);
  const double c      = Constants.c;

  const double ck_Hptau = c*k/(Hp*dtaudx);

  // Make the vector we are going to fill
  Vector y(Constants.n_ell_tot_full);

  //=============================================================================
  // Compute where in the y array each component belongs and where corresponding
  // components are located in the y_tc array
  // This is just an example of how to do it to make it easier
  // Feel free to organize the component any way you like
  //=============================================================================

  // Number of multipoles we have in the full regime
  const int n_ell_theta         = Constants.n_ell_theta;
  const int n_ell_thetap        = Constants.n_ell_thetap;
  const int n_ell_neutrinos     = Constants.n_ell_neutrinos;
  const bool polarization       = Constants.polarization;
  const bool neutrinos          = Constants.neutrinos;

  // Number of multipoles we have in the tight coupling regime
  const int n_ell_theta_tc      = Constants.n_ell_theta_tc;
  const int n_ell_neutrinos_tc  = Constants.n_ell_neutrinos_tc;

  // References to the tight coupling quantities
  const double &delta_cdm_tc    =  y_tc[Constants.ind_deltacdm_tc];
  const double &delta_b_tc      =  y_tc[Constants.ind_deltab_tc];
  const double &v_cdm_tc        =  y_tc[Constants.ind_vcdm_tc];
  const double &v_b_tc          =  y_tc[Constants.ind_vb_tc];
  const double &Phi_tc          =  y_tc[Constants.ind_Phi_tc];
  const double *Theta_tc        = &y_tc[Constants.ind_start_theta_tc];
  //const double *Nu_tc           = &y_tc[Constants.ind_start_nu_tc];

  // References to the quantities we are going to set
  double &delta_cdm       =  y[Constants.ind_deltacdm_tc];
  double &delta_b         =  y[Constants.ind_deltab_tc];
  double &v_cdm           =  y[Constants.ind_vcdm_tc];
  double &v_b             =  y[Constants.ind_vb_tc];
  double &Phi             =  y[Constants.ind_Phi_tc];
  double *Theta           = &y[Constants.ind_start_theta_tc];
  //double *Theta_p         = &y[Constants.ind_start_thetap_tc];
  //double *Nu              = &y[Constants.ind_start_nu_tc];

  //=============================================================================
  // TODO: fill in the initial conditions for the full equation system below
  // NB: remember that we have different number of multipoles in the two
  // regimes so be careful when assigning from the tc array
  //=============================================================================


  // SET: Scalar quantities (Gravitational potental, baryons and CDM)
  Phi = Phi_tc;
  delta_cdm = delta_cdm_tc;
  delta_b = delta_b_tc;
  v_cdm = v_cdm_tc;
  v_b = v_b_tc;

  // SET: Photon temperature perturbations (Theta_ell)
  double *Theta0 = Theta;
  double *Theta1 = Theta + 1;
  double *Theta2 = Theta + 2;

  *Theta0 = *Theta_tc;
  *Theta1 = *(Theta_tc + 1);
  *Theta2 = -20.0/45.0*ck_Hptau*(*Theta1);

  for (int l = 3; l < Constants.n_ell_theta; ++l)
  {
    double *Theta_l = Theta + l;
    double *Theta_lmin1 = Theta + (l - 1);
    *Theta_l = -l/(2.0*l + 1.0)*ck_Hptau*(*Theta_lmin1);
  }

  static bool print = true;

  if (print)
  {

    //std::cout << Phi << "\n";
    /*
    for (auto x: y)
    {
      std::cout << x << "\n";
    }
    */
  }

  // SET: Photon polarization perturbations (Theta_p_ell)
  if(polarization){
    // ...
    // ...
  }

  // SET: Neutrino perturbations (N_ell)
  if(neutrinos){
  }

  return y;
}

//====================================================
// The time when tight coupling end
//====================================================

double Perturbations::get_tight_coupling_time(const double k) const{
  double x_tight_coupling_end = 0.0;

  //=============================================================================
  // TODO: compute and return x for when tight coupling ends
  // Remember all the three conditions in Callin
  //=============================================================================

  const double c = Constants.c;
  double dx = (x_end - x_start)/n_x;

  for (int i = 0; i < n_x; ++i)
  {
    const double x = x_start + i*dx;

    const double Hp     = cosmo->Hp_of_x(x);
    const double dtaudx = rec->dtaudx_of_x(x);
    const double Xe     = rec->Xe_of_x(x);

    if ((abs(dtaudx) < 10.0*c*k/Hp) || (abs(dtaudx) < 10) || (Xe < 1))//(Xe < 0.999999))
    {
      x_tight_coupling_end = x;
      //std::cout << "x_tight = " << x_tight_coupling_end << "\n";
      break;
    }
  }
  return x_tight_coupling_end;
}

//====================================================
// After integrsating the perturbation compute the
// source function(s)
//====================================================
void Perturbations::compute_source_functions(){
  Utils::StartTiming("source");

  //=============================================================================
  // TODO: Make the x and k arrays to evaluate over and use to make the splines
  //=============================================================================
  // ...
  // ...
  Vector k_array;
  Vector x_array;

  // Make storage for the source functions (in 1D array to be able to pass it to the spline)
  Vector ST_array(k_array.size() * x_array.size());
  Vector SE_array(k_array.size() * x_array.size());

  // Compute source functions
  for(auto ix = 0; ix < x_array.size(); ix++){
    const double x = x_array[ix];
    for(auto ik = 0; ik < k_array.size(); ik++){
      const double k = k_array[ik];

      // NB: This is the format the data needs to be stored
      // in a 1D array for the 2D spline routine source(ix,ik) -> S_array[ix + nx * ik]
      const int index = ix + n_x * ik;

      //=============================================================================
      // TODO: Compute the source functions
      //=============================================================================
      // Fetch all the things we need...
      // const double Hp       = cosmo->Hp_of_x(x);
      // const double tau      = rec->tau_of_x(x);
      // ...
      // ...

      // Temperatur source
      ST_array[index] = 0.0;

      // Polarization source
      if(Constants.polarization){
        SE_array[index] = 0.0;
      }
    }
  }

  /*
  // Spline the source functions
  ST_spline.create (x_array, k_array, ST_array, "Source_Temp_x_k");
  if(Constants.polarization){
    SE_spline.create (x_array, k_array, SE_array, "Source_Pol_x_k");
  }
  */

  Utils::EndTiming("source");
}

//====================================================
// The right hand side of the perturbations ODE
// in the tight coupling regime
//====================================================

// Derivatives in the tight coupling regime
int Perturbations::rhs_tight_coupling_ode(double x, double k, const double *y, double *dydx){

  const double Hp           = cosmo->Hp_of_x(x);
  const double dHpdx        = cosmo->dHpdx_of_x(x);
  const double H0           = cosmo->H_of_x(0);
  const double Omega_gamma0 = cosmo->get_OmegaR(0);
  const double OmegaCDM0    = cosmo->get_OmegaCDM(0);
  const double OmegaB0      = cosmo->get_OmegaB(0);

  const double a            = exp(x);
  const double R            = 4.0*Omega_gamma0/(3.0*OmegaB0*a);

  const double dtaudx       = rec->dtaudx_of_x(x);
  const double ddtauddx     = rec->ddtauddx_of_x(x);

  //=============================================================================
  // Compute where in the y / dydx array each component belongs
  // This is just an example of how to do it to make it easier
  // Feel free to organize the component any way you like
  //=============================================================================

  // For integration of perturbations in tight coupling regime (Only 2 photon multipoles + neutrinos needed)
  const int n_ell_theta_tc      = Constants.n_ell_theta_tc;
  const int n_ell_neutrinos_tc  = Constants.n_ell_neutrinos_tc;
  const bool neutrinos          = Constants.neutrinos;

  // The different quantities in the y array
  const double &delta_cdm       =  y[Constants.ind_deltacdm_tc];
  const double &delta_b         =  y[Constants.ind_deltab_tc];
  const double &v_cdm           =  y[Constants.ind_vcdm_tc];
  const double &v_b             =  y[Constants.ind_vb_tc];
  const double &Phi             =  y[Constants.ind_Phi_tc];
  const double *Theta           = &y[Constants.ind_start_theta_tc];
  //const double *Nu              = &y[Constants.ind_start_nu_tc];

  // References to the quantities we are going to set in the dydx array
  double &ddelta_cdmdx    =  dydx[Constants.ind_deltacdm_tc];
  double &ddelta_bdx      =  dydx[Constants.ind_deltab_tc];
  double &dv_cdmdx        =  dydx[Constants.ind_vcdm_tc];
  double &dv_bdx          =  dydx[Constants.ind_vb_tc];
  double &dPhidx          =  dydx[Constants.ind_Phi_tc];
  double *dThetadx        = &dydx[Constants.ind_start_theta_tc];
  //double *dNudx           = &dydx[Constants.ind_start_nu_tc];

  //=============================================================================
  // TODO: fill in the expressions for all the derivatives
  //=============================================================================

  const double c = Constants.c;
  const double ck_Hp = c*k/Hp;

  const double Theta0 = *(Theta);
  const double Theta1 = *(Theta + 1);
  const double Theta2 = -20.0/45.0*ck_Hp/dtaudx*Theta1;

  // SET: Scalar quantities (Phi, delta, v, ...)
  double Psi = -Phi - 12.0*H0*H0/(c*c*k*k*a*a)*Omega_gamma0*Theta2;
  dPhidx = Psi - ck_Hp*ck_Hp/3.0*Phi + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm \
                                                            + OmegaB0/a*delta_b \
                                                            + 4*Omega_gamma0/(a*a)*Theta0);

  //std::cout << Phi << "\n";

  ddelta_cdmdx = ck_Hp*v_cdm - 3*dPhidx;
  ddelta_bdx = ck_Hp*v_b - 3*dPhidx;
  dv_cdmdx = -v_cdm - ck_Hp*Psi;

  //double dThetadx0 = -ck_Hp*Theta1 - dPhidx;
  double *dTheta0dx = dThetadx;
  double *dTheta1dx = dThetadx + 1;

  *dTheta0dx = -ck_Hp*Theta1 - dPhidx;

  double q = (-((1.0 - R)*dtaudx + (1.0 + R)*ddtauddx)*(3*Theta1 + v_b) - ck_Hp*Psi \
              + (1.0 - dHpdx/Hp)*ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*(*dTheta0dx))/((1.0 + R)*dtaudx + dHpdx/Hp - 1.0);

  dv_bdx = 1.0/(1.0 + R)*(-v_b - ck_Hp*Psi + R*(q + ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*Psi));

  //*dTheta0dx = -ck_Hp*Theta_1 - dPhidx;
  *dTheta1dx = 1.0/3.0*(q - dv_bdx);

  // SET: Neutrino mutlipoles (Nu_ell)
  if(neutrinos){
    // ...
    // ...
    // ...
  }

  return GSL_SUCCESS;
}

//====================================================
// The right hand side of the full ODE
//====================================================

int Perturbations::rhs_full_ode(double x, double k, const double *y, double *dydx){

  /*
  Look at delta_cdm and delta_b in my plots vs the plots from the project description.
  The deviation of delta_b from delta_cdm looks correct in the tight coupling regime, but
  not after recombination (when tight coupling has definitely ended).
  So looks like the solution of the full system is implemented wrongly.

  All the derivatives of the scalar quantities in the full system seemed to be correct.
  So there may be some mistake in the multipole derivatives.
  */

  //=============================================================================
  // Compute where in the y / dydx array each component belongs
  // This is just an example of how to do it to make it easier
  // Feel free to organize the component any way you like
  //=============================================================================

  // Index and number of the different quantities
  const int n_ell_theta         = Constants.n_ell_theta;
  const int n_ell_thetap        = Constants.n_ell_thetap;
  const int n_ell_neutrinos     = Constants.n_ell_neutrinos;
  const bool polarization       = Constants.polarization;
  const bool neutrinos          = Constants.neutrinos;

  // The different quantities in the y array
  const double &delta_cdm       =  y[Constants.ind_deltacdm];
  const double &delta_b         =  y[Constants.ind_deltab];
  const double &v_cdm           =  y[Constants.ind_vcdm];
  const double &v_b             =  y[Constants.ind_vb];
  const double &Phi             =  y[Constants.ind_Phi];
  const double *Theta           = &y[Constants.ind_start_theta];
  //const double *Theta_p         = &y[Constants.ind_start_thetap];
  //const double *Nu              = &y[Constants.ind_start_nu];

  // References to the quantities we are going to set in the dydx array
  double &ddelta_cdmdx    =  dydx[Constants.ind_deltacdm];
  double &ddelta_bdx      =  dydx[Constants.ind_deltab];
  double &dv_cdmdx        =  dydx[Constants.ind_vcdm];
  double &dv_bdx          =  dydx[Constants.ind_vb];
  double &dPhidx          =  dydx[Constants.ind_Phi];
  double *dThetadx        = &dydx[Constants.ind_start_theta];
  //double *dTheta_pdx      = &dydx[Constants.ind_start_thetap];
  //double *dNudx           = &dydx[Constants.ind_start_nu];

  // Cosmological/recombination parameters and variables
  const double Hp     = cosmo->Hp_of_x(x);
  const double H0     = cosmo->H_of_x(0);
  const double eta    = cosmo->eta_of_x(x);
  const double a      = exp(x);
  const double dtaudx = rec->dtaudx_of_x(x);

  const double Omega_gamma0 = cosmo->get_OmegaR(0);
  const double OmegaB0      = cosmo->get_OmegaB(0);
  const double OmegaCDM0    = cosmo->get_OmegaCDM(0);
  const double R            = 4.0/(3.0*OmegaB0*a);

  const double c = Constants.c;
  const double ck_Hp = c*k/Hp;



  //=============================================================================
  // TODO: fill in the expressions for all the derivatives
  //=============================================================================

  const double Theta0 = *(Theta);
  const double Theta1 = *(Theta + 1);
  const double Theta2 = *(Theta + 2);

  // SET: Scalar quantities (Phi, delta, v, ...)
  double Psi = -Phi - 12.0*(H0*H0/(c*c*k*k*a*a))*Omega_gamma0*Theta2;
  dPhidx = Psi - 1.0/3.0*ck_Hp*ck_Hp*Phi + 1.0/2.0*(H0*H0/(Hp*Hp))*(OmegaCDM0/a*delta_cdm + OmegaB0/a*delta_b \
                                                                  + 4*Omega_gamma0/(a*a)*Theta0);

  ddelta_cdmdx = ck_Hp*v_cdm - 3*dPhidx;
  ddelta_bdx   = ck_Hp*v_b - 3*dPhidx;

  dv_cdmdx     = -v_cdm - ck_Hp*Psi;
  dv_bdx       = -v_b - ck_Hp*Psi + dtaudx*R*(3*Theta1 + v_b);

  // SET: Photon multipoles (Theta_ell)
  double *dTheta0dx = dThetadx;
  double *dTheta1dx = dThetadx + 1;

  *dTheta0dx = -ck_Hp*Theta1 - dPhidx;
  *dTheta1dx = ck_Hp/3.0*Theta0 - 2.0*ck_Hp/3.0*Theta2 + ck_Hp/3.0*Psi + dtaudx*(Theta1 + 1.0/3.0*v_b);

  for (int l = 2; l < Constants.n_ell_theta; ++l)
  {
    double *dThetaldx         = dThetadx + l;
    const double Thetal      = *(Theta + l);
    const double Thetalmin1  = *(Theta + l - 1);

    if (l == Constants.n_ell_theta - 1)
    {
      *dThetaldx = ck_Hp*Thetalmin1 - c/(Hp*eta)*(l + 1.0)*Thetal + dtaudx*Thetal;
    }
    else
    {
      const double Thetalplus1 = *(Theta + l + 1);

      *dThetaldx = l*ck_Hp/(2.0*l + 1.0)*Thetalmin1 - (l + 1.0)/(2.0*l + 1.0)*ck_Hp*Thetalplus1 + \
                 + dtaudx*Thetal;

      if (l == 2)
      {
        *dThetaldx -= dtaudx/10.0*Thetal;
      }

    }
  }

  // SET: Photon polarization multipoles (Theta_p_ell)
  if(polarization){
    // ...
    // ...
    // ...
  }

  // SET: Neutrino mutlipoles (Nu_ell)
  if(neutrinos){
    // ...
    // ...
    // ...
  }

  return GSL_SUCCESS;
}

//====================================================
// Get methods
//====================================================

double Perturbations::get_delta_cdm(const double x, const double k) const{
  return delta_cdm_spline(x,k);
}
double Perturbations::get_delta_b(const double x, const double k) const{
  return delta_b_spline(x,k);
}
double Perturbations::get_v_cdm(const double x, const double k) const{
  return v_cdm_spline(x,k);
}
double Perturbations::get_v_b(const double x, const double k) const{
  return v_b_spline(x,k);
}
double Perturbations::get_Phi(const double x, const double k) const{
  return Phi_spline(x,k);
}

/*
double Perturbations::get_Psi(const double x, const double k) const{
  return Psi_spline(x,k);
}
double Perturbations::get_Pi(const double x, const double k) const{
  return Pi_spline(x,k);
}
double Perturbations::get_Source_T(const double x, const double k) const{
  return ST_spline(x,k);
}
double Perturbations::get_Source_E(const double x, const double k) const{
  return SE_spline(x,k);
}
*/

double Perturbations::get_Theta(const double x, const double k, const int ell) const{
  return Theta_spline[ell](x,k);
}

/*
double Perturbations::get_Theta_p(const double x, const double k, const int ell) const{
  return Theta_p_spline[ell](x,k);
}
double Perturbations::get_Nu(const double x, const double k, const int ell) const{
  return Nu_spline[ell](x,k);
}
*/

//====================================================
// Print some useful info about the class
//====================================================

void Perturbations::info() const{
  std::cout << "\n";
  std::cout << "Info about perturbations class:\n";
  std::cout << "x_start:       " << x_start                << "\n";
  std::cout << "x_end:         " << x_end                  << "\n";
  std::cout << "n_x:     " << n_x              << "\n";
  std::cout << "k_min (1/Mpc): " << k_min * Constants.Mpc  << "\n";
  std::cout << "k_max (1/Mpc): " << k_max * Constants.Mpc  << "\n";
  std::cout << "n_k:     " << n_k              << "\n";
  if(Constants.polarization)
    std::cout << "We include polarization\n";
  else
    std::cout << "We do not include polarization\n";
  if(Constants.neutrinos)
    std::cout << "We include neutrinos\n";
  else
    std::cout << "We do not include neutrinos\n";

  std::cout << "Information about the perturbation system:\n";
  std::cout << "ind_deltacdm:       " << Constants.ind_deltacdm         << "\n";
  std::cout << "ind_deltab:         " << Constants.ind_deltab           << "\n";
  std::cout << "ind_v_cdm:          " << Constants.ind_vcdm             << "\n";
  std::cout << "ind_v_b:            " << Constants.ind_vb               << "\n";
  std::cout << "ind_Phi:            " << Constants.ind_Phi              << "\n";
  std::cout << "ind_start_theta:    " << Constants.ind_start_theta      << "\n";
  std::cout << "n_ell_theta:        " << Constants.n_ell_theta          << "\n";
  if(Constants.polarization){
    std::cout << "ind_start_thetap:   " << Constants.ind_start_thetap   << "\n";
    std::cout << "n_ell_thetap:       " << Constants.n_ell_thetap       << "\n";
  }
  if(Constants.neutrinos){
    std::cout << "ind_start_nu:       " << Constants.ind_start_nu       << "\n";
    std::cout << "n_ell_neutrinos     " << Constants.n_ell_neutrinos    << "\n";
  }
  std::cout << "n_ell_tot_full:     " << Constants.n_ell_tot_full       << "\n";

  std::cout << "Information about the perturbation system in tight coupling:\n";
  std::cout << "ind_deltacdm:       " << Constants.ind_deltacdm_tc      << "\n";
  std::cout << "ind_deltab:         " << Constants.ind_deltab_tc        << "\n";
  std::cout << "ind_v_cdm:          " << Constants.ind_vcdm_tc          << "\n";
  std::cout << "ind_v_b:            " << Constants.ind_vb_tc            << "\n";
  std::cout << "ind_Phi:            " << Constants.ind_Phi_tc           << "\n";
  std::cout << "ind_start_theta:    " << Constants.ind_start_theta_tc   << "\n";
  std::cout << "n_ell_theta:        " << Constants.n_ell_theta_tc       << "\n";
  if(Constants.neutrinos){
    std::cout << "ind_start_nu:       " << Constants.ind_start_nu_tc    << "\n";
    std::cout << "n_ell_neutrinos     " << Constants.n_ell_neutrinos_tc << "\n";
  }
  std::cout << "n_ell_tot_tc:       " << Constants.n_ell_tot_tc         << "\n";
  std::cout << std::endl;
}

//====================================================
// Output some results to file for a given value of k
//====================================================

void Perturbations::output(const double k, const std::string filename) const{
  std::ofstream fp(filename.c_str());
  const int npts = 5000;
  auto x_array = Utils::linspace(x_start, x_end, npts);
  auto print_data = [&] (const double x) {
    double arg = k * (cosmo->eta_of_x(0.0) - cosmo->eta_of_x(x));
    fp << x                  << " ";
    fp << get_delta_cdm(x, k)<< " ";
    fp << get_delta_b(x, k)  << " ";
    fp << get_v_cdm(x, k)    << " ";
    fp << get_v_b(x, k)      << " ";
    fp << get_Theta(x,k,0)   << " ";
    fp << get_Theta(x,k,1)   << " ";
    fp << get_Theta(x,k,2)   << " ";
    fp << get_Phi(x,k)       << " ";
    //fp << get_Psi(x,k)       << " ";
    //fp << get_Pi(x,k)        << " ";
    //fp << get_Source_T(x,k)  << " ";
    //fp << get_Source_T(x,k) * Utils::j_ell(5,   arg)           << " ";
    //fp << get_Source_T(x,k) * Utils::j_ell(50,  arg)           << " ";
    //fp << get_Source_T(x,k) * Utils::j_ell(500, arg)           << " ";
    fp << "\n";
  };
  std::for_each(x_array.begin(), x_array.end(), print_data);
}
