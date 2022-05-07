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
  // Vectors containing splines for the photon, neutrino and polarization multipoles
  Theta_spline   = std::vector<Spline2D>(Constants.n_ell_theta);
  Nu_spline      = std::vector<Spline2D>(Constants.n_ell_neutrinos);
  Theta_p_spline = std::vector<Spline2D>(Constants.n_ell_thetap);

  const double Omega_nu0    = cosmo->get_OmegaNu(0);
  const double Omega_gamma0 = cosmo->get_OmegaR(0);

  f_nu = Omega_nu0/(Omega_gamma0 + Omega_nu0);
}

//====================================================
// Do all the solving
//====================================================

void Perturbations::solve(){

  // Integrate all the perturbation equations and spline the result
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

  // Generating logarithmically spaced k-values by first generating
  // linearly spaced exponents
  Vector exponents = Utils::linspace(log(k_min), log(k_max), n_k);

  Vector k_array(n_k);

  for (int i = 0; i < n_k; ++i)
  {
    k_array[i] = exp(exponents[i]);
  }

  // Initializing the vectors containing the values for all the perturbation variables
  Vector delta_cdm(n_x*n_k);
  Vector delta_b(n_x*n_k);
  Vector v_cdm(n_x*n_k);
  Vector v_b(n_x*n_k);
  Vector Phi(n_x*n_k);
  Vector Psi(n_x*n_k);

  Vector2D Thetas(Constants.n_ell_theta, Vector(n_x*n_k, 0.0));
  Vector2D Nu(Constants.n_ell_neutrinos, Vector(n_x*n_k, 0.0));
  Vector2D Theta_p(Constants.n_ell_thetap, Vector(n_x*n_k, 0.0));

  // Vector for storing the indices of x_array at which tight coupling ends for the
  // different k-values
  Vector indices_end_tight(n_k);

  #pragma omp parallel for schedule(dynamic, 8)
  // Loop over all wavenumbers
  for(int ik = 0; ik < n_k; ++ik)
  {
    // Matrix that will be filled with the values for all the perturbation variables
    // as functions of x (for this particular k-value)
    Vector2D y(n_x, Vector(Constants.n_ell_tot_full, 0.0));

    // Current value of k
    double k = k_array[ik];

    // x-value at which tight coupling should end
    double x_end_tight = get_tight_coupling_time(k);

    // Set up initial conditions in the tight coupling regime
    auto y_tight_coupling_ini = set_ic(x_start, k);

    // The tight coupling ODE system
    ODEFunction dydx_tight_coupling = [&](double x, const double *y, double *dydx){
      return rhs_tight_coupling_ode(x, k, y, dydx);
    };

    // Index of x_array corresponding to the end of tight coupling
    const int index_end_tight = std::lower_bound(x_array.begin(), x_array.end(), x_end_tight) - x_array.begin();
    indices_end_tight[ik] = index_end_tight;

    // Vector containing x-values for the tight coupling regime
    Vector x_tight;
    x_tight = Vector(x_array.begin(), x_array.begin() + index_end_tight);

    // Defining ODESolver for the tight coupling regime and solving the system in that regime
    ODESolver tc_ode;
    tc_ode.solve(dydx_tight_coupling, x_tight, y_tight_coupling_ini);
    auto y_tight_coupling = tc_ode.get_data();

    // Values at end of tight coupling. Initial conditions for full system
    Vector y_end_tight = y_tight_coupling.back();

    // Vector containing x-values for the full system
    Vector x_after_tight;
    // Initial conditions for full system are at x = x_array.begin() + index_end_tight - 1
    x_after_tight = Vector(x_array.begin() + index_end_tight - 1, x_array.end());

    // Set up initial conditions for full system
    auto y_full_ini = set_ic_after_tight_coupling(y_end_tight, x_after_tight[0], k);

    // The full ODE system
    ODEFunction dydx_full = [&](double x, const double *y, double *dydx){
      return rhs_full_ode(x, k, y, dydx);
    };

    // Integrate from x_end_tight -> x_end
    ODESolver full_ode;
    full_ode.solve(dydx_full, x_after_tight, y_full_ini);
    auto y_full = full_ode.get_data();

    //=======================================================================================
    // Inserting the computed perturbation variables into their respective vectors,
    // and calculating the values for the higher order multipoles that were not
    // computed during tight coupling.
    //=======================================================================================

    // Constants used to calculate higher order multipoles and Psi
    const double H0           = cosmo->H_of_x(0);
    const double Omega_gamma0 = cosmo->get_OmegaR(0);
    const double Omega_nu0    = cosmo->get_OmegaNu(0);
    const double c            = Constants.c;

    for (int ix = 0; ix < n_x; ++ix)
    {
      const int index = ix + n_x*ik;

      const double x = x_array[ix];
      const double a = exp(x);

      // During tight coupling we have not computed higher order multipoles, so need to do that here
      // Also store all the variables that have been computed
      if (ix < index_end_tight - 1)
      {
        delta_cdm[index] = y_tight_coupling[ix][Constants.ind_deltacdm];
        delta_b[index]   = y_tight_coupling[ix][Constants.ind_deltab];
        v_cdm[index]     = y_tight_coupling[ix][Constants.ind_vcdm];
        v_b[index]       = y_tight_coupling[ix][Constants.ind_vb];
        Phi[index]       = y_tight_coupling[ix][Constants.ind_Phi];
        Thetas[0][index] = y_tight_coupling[ix][Constants.ind_start_theta_tc];
        Thetas[1][index] = y_tight_coupling[ix][Constants.ind_start_theta_tc + 1];

        // Needed for the calculation of the higher order multipoles
        const double Hp     = cosmo->Hp_of_x(x);
        const double dtaudx = rec->dtaudx_of_x(x);
        const double ck_Hp  = c*k/Hp;

        // Different formulas for Theta_2 depending on if we include polarization
        if (Constants.polarization)
        {
          Thetas[2][index] = -8.0/15.0*ck_Hp/dtaudx*Thetas[1][index];
        }
        else
        {
          Thetas[2][index] = -20.0/45.0*ck_Hp/dtaudx*Thetas[1][index];
        }

        for (int l = 3; l < Constants.n_ell_theta; ++l)
        {

          Thetas[l][index] = -l/(2.0*l + 1.0)*ck_Hp/dtaudx*Thetas[l - 1][index];
        }

        // We don't calculate any polarization multipoles during tight coupling,
        // so need to calculate all of them here (if polarization is included)
        if (Constants.polarization)
        {
          Theta_p[0][index] = 5.0/4.0*Thetas[2][index];
          Theta_p[1][index] = -ck_Hp/(4.0*dtaudx)*Thetas[2][index];
          Theta_p[2][index] = 1.0/4.0*Thetas[2][index];

          for (int l = 3; l < Constants.n_ell_thetap; ++l)
          {
            Theta_p[l][index] = -l/(2.0*l + 1.0)*ck_Hp/dtaudx*Theta_p[l - 1][index];
          }
        }
      }
      // After tight coupling:
      else
      {
        // For indexing y_full
        const int full_ix = ix - (index_end_tight - 1);

        delta_cdm[index] = y_full[full_ix][Constants.ind_deltacdm];
        delta_b[index]   = y_full[full_ix][Constants.ind_deltab];
        v_cdm[index]     = y_full[full_ix][Constants.ind_vcdm];
        v_b[index]       = y_full[full_ix][Constants.ind_vb];
        Phi[index]       = y_full[full_ix][Constants.ind_Phi];

        // Temperature multipoles
        for (int l = 0; l < Constants.n_ell_theta; ++l)
        {
          Thetas[l][index] = y_full[full_ix][Constants.ind_start_theta + l];
        }

        // Storing neutrino perturbations if they are included
        if (Constants.neutrinos)
        {
          for (int l = 0; l < Constants.n_ell_neutrinos; ++l)
          {
            Nu[l][index] = y_full[full_ix][Constants.ind_start_nu + l];
          }
        }

        // Storing polarization perturbations if they are included
        if (Constants.polarization)
        {
          for (int l = 0; l < Constants.n_ell_thetap; ++l)
          {
            Theta_p[l][index] = y_full[full_ix][Constants.ind_start_thetap + l];
          }
        }
      }

      // Calculating and storing Psi
      if (Constants.neutrinos)
      {
        Psi[index] = -Phi[index] - 12.0*H0*H0/(c*c*k*k*a*a)*(Omega_gamma0*Thetas[2][index] \
                                                           + Omega_nu0*Nu[2][index]);
      }
      else
      {
        Psi[index] = -Phi[index] - 12.0*H0*H0/(c*c*k*k*a*a)*Omega_gamma0*Thetas[2][index];
      }
    }
  }

  Utils::EndTiming("integrateperturbation");

  //=========================================================================================
  // Creating splines. First for the variables for which we have the complete time-evolutions
  // (meaning everything except the higher order multipoles)
  //=========================================================================================
  delta_cdm_spline.create(x_array, k_array, delta_cdm);
  delta_b_spline.create(x_array, k_array, delta_b);
  v_cdm_spline.create(x_array, k_array, v_cdm);
  v_b_spline.create(x_array, k_array, v_b);
  Phi_spline.create(x_array, k_array, Phi);
  Psi_spline.create(x_array, k_array, Psi);

  // Creating splines for photon multipoles and polarization and neutrino multipoles (if included)

  for (int l = 0; l < Constants.n_ell_theta; ++l)
  {
    std::string spline_name = "theta_" + std::to_string(l) + "_spline";
    Theta_spline[l].create(x_array, k_array, Thetas[l], spline_name);
  }

  if (Constants.neutrinos)
  {
    for (int l = 0; l < Constants.n_ell_neutrinos; ++l)
    {
      std::string spline_name = "nu_" + std::to_string(l) + "_spline";
      Nu_spline[l].create(x_array, k_array, Nu[l], spline_name);
    }
  }

  if (Constants.polarization)
  {
    for (int l = 0; l < Constants.n_ell_thetap; ++l)
    {
      std::string spline_name = "theta_p_" + std::to_string(l) + "_spline";
      Theta_p_spline[l].create(x_array, k_array, Theta_p[l], spline_name);
    }
  }
}

//====================================================
// Set IC at the start of the run (this is in the
// tight coupling regime)
//====================================================
Vector Perturbations::set_ic(const double x, const double k) const{

  // Constants and variables needed in the expression for the initial
  // conditions in the tight coupling regime
  const double c         = Constants.c;
  const double Hp        = cosmo->Hp_of_x(x);
  const double H0        = cosmo->H_of_x(0);
  const double dtaudx    = rec->dtaudx_of_x(x);
  const double a         = exp(x);
  const double Omega_Nu0 = cosmo->get_OmegaNu(0);

  // Used multiple times, so to avoid having to calculate this more times than necessary
  const double ck_Hp = c*k/Hp;

  // The vector we are going to fill
  Vector y_tc(Constants.n_ell_tot_tc);

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
  double *Nu;

  if (neutrinos)
  {
    Nu = &y_tc[Constants.ind_start_nu_tc];
  }

  // Defining initial conditions in tight coupling
  double Psi = -1.0/(3.0/2.0 + 2.0/5.0*f_nu);
  Phi = -(1 + 2.0/5.0*f_nu)*Psi;

  delta_cdm = -3.0/2.0*Psi;
  delta_b = delta_cdm;

  v_cdm = -c*k/(2.0*Hp)*Psi;
  v_b = v_cdm;

  double *Theta_0 = Theta;
  double *Theta_1 = Theta + 1;

  *Theta_0 = -1.0/2.0*Psi;
  *Theta_1 = ck_Hp/6.0*Psi;

  // Calculating initial conditions for the neutrino multipoles
  if (neutrinos)
  {
    double *Nu0 = Nu;
    double *Nu1 = Nu + 1;
    double *Nu2 = Nu + 2;

    *Nu0 = -1.0/2.0*Psi;
    *Nu1 = ck_Hp/6.0*Psi;
    *Nu2 = -c*c*k*k*a*a/(12.0*H0*H0*Omega_Nu0)*(Phi + Psi);

    for (int l = 3; l < Constants.n_ell_neutrinos_tc; ++l)
    {
      const double Nu_lmin1 = *(Nu + l -1);
      double *Nu_l = Nu + l;

      *Nu_l = ck_Hp/(2.0*l + 1.0)*Nu_lmin1;
    }
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

  // Constants and variables needed in the expression for the initial
  // conditions in the full system
  const double Hp     = cosmo->Hp_of_x(x);
  const double dtaudx = rec->dtaudx_of_x(x);
  const double c      = Constants.c;

  const double ck_Hp    = c*k/Hp;
  const double ck_Hptau = ck_Hp/dtaudx;

  // Make the vector we are going to fill
  Vector y(Constants.n_ell_tot_full);

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
  const double *Nu_tc;

  // References to the quantities we are going to set
  double &delta_cdm       =  y[Constants.ind_deltacdm_tc];
  double &delta_b         =  y[Constants.ind_deltab_tc];
  double &v_cdm           =  y[Constants.ind_vcdm_tc];
  double &v_b             =  y[Constants.ind_vb_tc];
  double &Phi             =  y[Constants.ind_Phi_tc];
  double *Theta           = &y[Constants.ind_start_theta_tc];
  double *Nu;
  double *Theta_p;

  if (neutrinos)
  {
    Nu_tc = &y_tc[Constants.ind_start_nu_tc];
    Nu    = &y[Constants.ind_start_nu];
  }

  if (polarization)
  {
    Theta_p    = &y[Constants.ind_start_thetap];
  }

  // Initial conditions in the full system.
  Phi = Phi_tc;
  delta_cdm = delta_cdm_tc;
  delta_b = delta_b_tc;
  v_cdm = v_cdm_tc;
  v_b = v_b_tc;

  double *Theta0 = Theta;
  double *Theta1 = Theta + 1;
  double *Theta2 = Theta + 2;

  // Initial conditions for higher order multipoles need to be calculated from
  // monopole and dipole.
  *Theta0 = *Theta_tc;
  *Theta1 = *(Theta_tc + 1);

  // Calculating polarization multipoles at end of tight coupling
  if (polarization)
  {
    *Theta2 = -8.0/15.0*ck_Hptau*(*Theta1);

    double *Theta_p0 = Theta_p;
    double *Theta_p1 = Theta_p + 1;
    double *Theta_p2 = Theta_p + 2;

    *Theta_p0 = 5.0/4.0*(*Theta2);
    *Theta_p1 = -ck_Hp/(4.0*dtaudx)*(*Theta2);
    *Theta_p2 = 1.0/4.0*(*Theta2);

    for (int l = 3; l < Constants.n_ell_thetap; ++l)
    {
      const double Theta_plmin1 = *(Theta_p + l - 1);
      double *Theta_p_l = Theta_p + l;

      *Theta_p_l = -l/(2.0*l + 1.0)*ck_Hptau*Theta_plmin1;
    }
  }
  else
  {
    // Different formula for Theta_2 with polarization
    *Theta2 = -20.0/45.0*ck_Hptau*(*Theta1);
  }

  // Calculating higher order multipoles that were not calculated in
  // the tight coupling regime
  for (int l = 3; l < Constants.n_ell_theta; ++l)
  {
    double *Theta_l = Theta + l;
    double *Theta_lmin1 = Theta + (l - 1);
    *Theta_l = -l/(2.0*l + 1.0)*ck_Hptau*(*Theta_lmin1);
  }

  if (neutrinos)
  {
    for (int l = 0; l < Constants.n_ell_neutrinos; ++l)
    {
      const double Nu_tc_l = *(Nu_tc + l);
      double *Nu_l = Nu + l;

      *Nu_l = Nu_tc_l;
    }
  }


  return y;
}

//====================================================
// The time when tight coupling end
//====================================================

double Perturbations::get_tight_coupling_time(const double k) const{
  double x_tight_coupling_end = 0.0;

  const double c = Constants.c;
  double dx = (x_end - x_start)/n_x; // Distance between x-values in x_array

  // Looping through each x-value to find when tight coupling ends
  for (int i = 0; i < n_x; ++i)
  {
    const double x = x_start + i*dx;

    const double Hp     = cosmo->Hp_of_x(x);
    const double dtaudx = rec->dtaudx_of_x(x);
    const double Xe     = rec->Xe_of_x(x);

    // When the optical depth is too small or we reach recombination then
    // tight coupling should end
    if ((abs(dtaudx) < 10.0*c*k/Hp) || (abs(dtaudx) < 10) || (Xe < 1))
    {
      x_tight_coupling_end = x;
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

  Vector x_array = Utils::linspace(x_start, x_end, n_x);

  // Again generating logarithmically spaced k-values

  const int nk = 1000;

  Vector exponents = Utils::linspace(log(k_min), log(k_max), nk);

  Vector k_array(nk);

  for (int i = 0; i < nk; ++i)
  {
    k_array[i] = exp(exponents[i]);
  }

  // Make storage for the source function (in 1D array to be able to pass it to the spline)
  Vector ST_array(k_array.size() * x_array.size());

  // Compute source function
  for(auto ix = 0; ix < x_array.size(); ix++){
    const double x = x_array[ix];

    #pragma omp parallel for schedule(dynamic, 8)
    for(auto ik = 0; ik < k_array.size(); ik++){
      const double k = k_array[ik];

      // NB: This is the format the data needs to be stored
      // in a 1D array for the 2D spline routine source(ix,ik) -> S_array[ix + nx * ik]
      const int index = ix + n_x * ik;

      // Fetch all the things we need...
      const double H0        = cosmo->H_of_x(0);
      const double Hp        = cosmo->Hp_of_x(x);
      const double dHpdx     = cosmo->dHpdx_of_x(x);
      const double ddHpddx   = cosmo->ddHpddx_of_x(x);
      const double tau       = rec->tau_of_x(x);
      const double dtaudx    = rec->dtaudx_of_x(x);
      const double ddtauddx  = rec->ddtauddx_of_x(x);
      const double g_tilde   = rec->g_tilde_of_x(x);
      const double dgdx      = rec->dgdx_tilde_of_x(x);
      const double ddgddx    = rec->ddgddx_tilde_of_x(x);
      const double c         = Constants.c;
      const double a         = exp(x);
      const double OmegaR0   = cosmo->get_OmegaR(0);
      const double OmegaCDM0 = cosmo->get_OmegaCDM(0);
      const double OmegaB0   = cosmo->get_OmegaB(0);
      const double Xe     = rec->Xe_of_x(x);

      const double ck_Hp     = c*k/Hp;

      const double Theta0    = get_Theta(x, k, 0);
      const double Theta1    = get_Theta(x, k, 1);
      const double Theta2    = get_Theta(x, k, 2);
      const double Theta3    = get_Theta(x, k, 3);
      const double Theta4    = get_Theta(x, k, 4);
      const double Phi       = get_Phi(x, k);
      const double Psi       = get_Psi(x, k);
      const double v_b       = get_v_b(x, k);
      const double delta_b   = get_delta_b(x, k);
      const double delta_cdm = get_delta_cdm(x, k);


      const double R = 4.0/3.0*OmegaR0/(OmegaB0*a);

      /*
      const double dPhidx    = Psi - ck_Hp*ck_Hp/3.0*Phi \
                             + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm + OmegaB0/a*delta_b \
                             + 4*OmegaR0/(a*a)*Theta0);

      double dTheta2dx;
      double ddTheta2ddx;
      double dv_bdx;

      // Different formulas for v_b and higher order photon multipoles
      // during tight coupling
      if ((abs(dtaudx) >= 10.0*ck_Hp) && (abs(dtaudx) >= 10) && (Xe >= 1))
      {
        const double ck_Hptau = ck_Hp/dtaudx;
        const double dTheta0dx = -ck_Hp*Theta1 - dPhidx;

        double q = (-((1.0 - R)*dtaudx + (1.0 + R)*ddtauddx)*(3*Theta1 + v_b) - ck_Hp*Psi \
        + (1.0 - dHpdx/Hp)*ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*(dTheta0dx))/((1.0 + R)*dtaudx + dHpdx/Hp - 1.0);

        dv_bdx = 1.0/(1.0 + R)*(-v_b - ck_Hp*Psi + R*(q + ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*Psi));

        const double dTheta1dx = 1.0/3.0*(q - dv_bdx);

        dTheta2dx = 20.0/45.0*ck_Hptau*(-dTheta1dx + dHpdx/Hp*Theta1 + ddtauddx/dtaudx*Theta1);

        //ddTheta1ddx = Theta_spline[1].deriv_xx(x, k);

        ddTheta2ddx = Theta_spline[2].deriv_xx(x, k);

      }
      else
      {

        dTheta2dx              = 2.0/5.0*ck_Hp*Theta1 - 3.0/5.0*ck_Hp*Theta3 + 9.0/10.0*dtaudx*Theta2;

        const double dTheta1dx = ck_Hp/3.0*Theta0 - 2.0/3.0*ck_Hp*Theta2 + ck_Hp/3.0*Psi \
                               + dtaudx*(Theta1 + 1.0/3.0*v_b);

        const double dTheta3dx = 3.0/7.0*ck_Hp*Theta2 - 4.0/7.0*ck_Hp*Theta4 + dtaudx*Theta3;
        dv_bdx                 = -v_b - ck_Hp*Psi + dtaudx*R*(3.0*Theta1 + v_b);

        ddTheta2ddx = 2.0/5.0*ck_Hp*(dTheta1dx - dHpdx/Hp*Theta1) \
                    - 3.0/5.0*ck_Hp*(dTheta3dx - dHpdx/Hp*Theta3) \
                    + 9.0/10.0*(dTheta2dx*dtaudx + Theta2*ddtauddx);
                    //+ 3.0/10.0*(dTheta2dx*dtaudx + Theta2*ddtauddx);
      }

      const double dPsidx    = -dPhidx - 12.0*H0*H0/(c*c*k*k*a*a)*OmegaR0*(dTheta2dx - 2*Theta2);
      //const double dPsidx = Psi_spline.deriv_x(x, k);
      */

      const double dPsidx      = Psi_spline.deriv_x(x, k);
      const double dPhidx      = Phi_spline.deriv_x(x, k);
      const double dTheta2dx   = Theta_spline[2].deriv_x(x, k);
      const double ddTheta2ddx = Theta_spline[2].deriv_xx(x, k);
      const double dv_bdx      = v_b_spline.deriv_x(x, k);

      const double last_term_1   = g_tilde*Theta2*(dHpdx*dHpdx + Hp*ddHpddx);
      const double last_term_2   = 3*Hp*dHpdx*(dgdx*Theta2 + g_tilde*dTheta2dx);
      const double last_term_3   = Hp*Hp*(ddgddx*Theta2 + 2*dgdx*dTheta2dx + g_tilde*ddTheta2ddx);

      const double first_term  = g_tilde*(Theta0 + Psi + 1.0/4.0*Theta2);
      const double second_term = exp(-tau)*(dPsidx - dPhidx);
      const double third_term  = -1.0/(c*k)*(dHpdx*g_tilde*v_b + Hp*dgdx*v_b + Hp*g_tilde*dv_bdx);
      const double fourth_term = 3.0/(4.0*c*c*k*k)*(last_term_1 + last_term_2 + last_term_3);

      ST_array[index] = first_term + second_term + third_term + fourth_term;

      }
    }

  // Spline the source function
  ST_spline.create(x_array, k_array, ST_array, "Source_Temp_x_k");

  Utils::EndTiming("source");
}

//====================================================
// The right hand side of the perturbations ODE
// in the tight coupling regime
//====================================================

// Derivatives in the tight coupling regime
int Perturbations::rhs_tight_coupling_ode(double x, double k, const double *y, double *dydx){

  // Cosmological and recombination quantites needed in the ODE system
  const double Hp           = cosmo->Hp_of_x(x);
  const double dHpdx        = cosmo->dHpdx_of_x(x);
  const double H0           = cosmo->H_of_x(0);
  const double eta          = cosmo->eta_of_x(x);
  const double Omega_gamma0 = cosmo->get_OmegaR(0);
  const double Omega_Nu0    = cosmo->get_OmegaNu(0);
  const double OmegaCDM0    = cosmo->get_OmegaCDM(0);
  const double OmegaB0      = cosmo->get_OmegaB(0);

  const double a            = exp(x);
  const double R            = 4.0*Omega_gamma0/(3.0*OmegaB0*a);

  const double dtaudx       = rec->dtaudx_of_x(x);
  const double ddtauddx     = rec->ddtauddx_of_x(x);

  const double c = Constants.c;
  const double ck_Hp = c*k/Hp;

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
  const double *Nu;

  // References to the quantities we are going to set in the dydx array
  double &ddelta_cdmdx    =  dydx[Constants.ind_deltacdm_tc];
  double &ddelta_bdx      =  dydx[Constants.ind_deltab_tc];
  double &dv_cdmdx        =  dydx[Constants.ind_vcdm_tc];
  double &dv_bdx          =  dydx[Constants.ind_vb_tc];
  double &dPhidx          =  dydx[Constants.ind_Phi_tc];
  double *dThetadx        = &dydx[Constants.ind_start_theta_tc];
  double *dNudx;

  if (neutrinos)
  {
    Nu = &y[Constants.ind_start_nu_tc];
    dNudx = &dydx[Constants.ind_start_nu_tc];
  }

  const double Theta0 = *(Theta);
  const double Theta1 = *(Theta + 1);

  double Theta2;

  if (Constants.polarization)
  {
    Theta2 = -8.0/15.0*ck_Hp/dtaudx*Theta1;
  }
  else
  {
    Theta2 = -20.0/45.0*ck_Hp/dtaudx*Theta1;
  }

  double Psi;

  // Calculating Psi, dPhidx and derivatives of the neutrino multipoles
  // when neutrinos are included
  if (neutrinos)
  {
    const double Nu0 = *Nu;
    const double Nu1 = *(Nu + 1);
    const double Nu2 = *(Nu + 2);

    Psi = -Phi - 12.0*H0*H0/(c*c*k*k*a*a)*(Omega_gamma0*Theta2 + Omega_Nu0*Nu2);
    dPhidx = Psi - ck_Hp*ck_Hp/3.0*Phi + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm \
                                                              + OmegaB0/a*delta_b \
                                                              + 4*Omega_gamma0/(a*a)*Theta0\
                                                              + 4*Omega_Nu0/(a*a)*Nu0);

    double *dNu0dx = dNudx;
    double *dNu1dx = dNudx + 1;

    *dNu0dx = -ck_Hp*Nu1 - dPhidx;
    *dNu1dx = ck_Hp/3.0*Nu0 - 2.0/3.0*ck_Hp*Nu2 + ck_Hp/3.0*Psi;

    for (int l = 2; l < Constants.n_ell_neutrinos_tc; ++l)
    {
      const double Nu_lmin1 = *(Nu + l - 1);
      double *dNudx_l = dNudx + l;

      if (l == Constants.n_ell_neutrinos_tc - 1)
      {
        const double Nu_l = *(Nu + l);
        *dNudx_l = ck_Hp*Nu_lmin1 - c/(Hp*eta)*(l + 1.0)*Nu_l;
      }

      else
      {
        const double Nu_lplus1 = *(Nu + l + 1);
        *dNudx_l = ck_Hp*l/(2.0*l + 1.0)*Nu_lmin1 - ck_Hp*(l + 1.0)/(2.0*l + 1.0)*Nu_lplus1;
      }
    }
  }

  // Psi and dPhidx without neutrinos
  else
  {
    Psi = -Phi - 12.0*H0*H0/(c*c*k*k*a*a)*(Omega_gamma0*Theta2);
    dPhidx = Psi - ck_Hp*ck_Hp/3.0*Phi + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm \
                                                              + OmegaB0/a*delta_b \
                                                              + 4*Omega_gamma0/(a*a)*Theta0);
  }



  ddelta_cdmdx = ck_Hp*v_cdm - 3*dPhidx;
  ddelta_bdx = ck_Hp*v_b - 3*dPhidx;
  dv_cdmdx = -v_cdm - ck_Hp*Psi;

  double *dTheta0dx = dThetadx;
  double *dTheta1dx = dThetadx + 1;

  *dTheta0dx = -ck_Hp*Theta1 - dPhidx;

  double q = (-((1.0 - R)*dtaudx + (1.0 + R)*ddtauddx)*(3*Theta1 + v_b) - ck_Hp*Psi \
              + (1.0 - dHpdx/Hp)*ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*(*dTheta0dx))/((1.0 + R)*dtaudx + dHpdx/Hp - 1.0);

  dv_bdx = 1.0/(1.0 + R)*(-v_b - ck_Hp*Psi + R*(q + ck_Hp*(-Theta0 + 2*Theta2) - ck_Hp*Psi));

  *dTheta1dx = 1.0/3.0*(q - dv_bdx);

  return GSL_SUCCESS;
}

//====================================================
// The right hand side of the full ODE
//====================================================

int Perturbations::rhs_full_ode(double x, double k, const double *y, double *dydx){

  // Cosmological/recombination parameters and variables
  const double Hp     = cosmo->Hp_of_x(x);
  const double H0     = cosmo->H_of_x(0);
  const double eta    = cosmo->eta_of_x(x);
  const double a      = exp(x);
  const double dtaudx = rec->dtaudx_of_x(x);

  const double Omega_gamma0 = cosmo->get_OmegaR(0);
  const double Omega_Nu0    = cosmo->get_OmegaNu(0);
  const double OmegaB0      = cosmo->get_OmegaB(0);
  const double OmegaCDM0    = cosmo->get_OmegaCDM(0);
  const double R            = 4.0*Omega_gamma0/(3.0*OmegaB0*a);

  const double c = Constants.c;
  const double ck_Hp = c*k/Hp;

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
  const double *Nu;
  const double *Theta_p;

  // References to the quantities we are going to set in the dydx array
  double &ddelta_cdmdx    =  dydx[Constants.ind_deltacdm];
  double &ddelta_bdx      =  dydx[Constants.ind_deltab];
  double &dv_cdmdx        =  dydx[Constants.ind_vcdm];
  double &dv_bdx          =  dydx[Constants.ind_vb];
  double &dPhidx          =  dydx[Constants.ind_Phi];
  double *dThetadx        = &dydx[Constants.ind_start_theta];
  double *dNudx;
  double *dTheta_pdx;

  if (neutrinos)
  {
    Nu = &y[Constants.ind_start_nu];
    dNudx = &dydx[Constants.ind_start_nu];
  }

  const double Theta0 = *(Theta);
  const double Theta1 = *(Theta + 1);
  const double Theta2 = *(Theta + 2);

  double Pi;

  if (polarization)
  {
    Theta_p = &y[Constants.ind_start_thetap];
    dTheta_pdx = &dydx[Constants.ind_start_thetap];

    const double Theta_p0 = *Theta_p;
    const double Theta_p2 = *(Theta_p + 2);

    Pi = Theta2 + Theta_p0 + Theta_p2;
  }
  else
  {
    Pi = Theta2;
  }

  double Psi;

  if (neutrinos)
  {
    const double Nu0 = *Nu;
    const double Nu1 = *(Nu + 1);
    const double Nu2 = *(Nu + 2);

    Psi = -Phi - 12.0*H0*H0/(c*c*k*k*a*a)*(Omega_gamma0*Theta2 + Omega_Nu0*Nu2);
    dPhidx = Psi - ck_Hp*ck_Hp/3.0*Phi + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm \
                                                              + OmegaB0/a*delta_b \
                                                              + 4*Omega_gamma0/(a*a)*Theta0\
                                                              + 4*Omega_Nu0/(a*a)*Nu0);

    double *dNu0dx = dNudx;
    double *dNu1dx = dNudx + 1;

    *dNu0dx = -ck_Hp*Nu1 - dPhidx;
    *dNu1dx = ck_Hp/3.0*Nu0 - 2.0/3.0*ck_Hp*Nu2 + ck_Hp/3.0*Psi;

    for (int l = 2; l < Constants.n_ell_neutrinos_tc; ++l)
    {
      const double Nu_lmin1 = *(Nu + l - 1);
      double *dNudx_l = dNudx + l;

      if (l == Constants.n_ell_neutrinos_tc - 1)
      {
        const double Nu_l = *(Nu + l);
        *dNudx_l = ck_Hp*Nu_lmin1 - c/(Hp*eta)*(l + 1.0)*Nu_l;
      }

      else
      {
        const double Nu_lplus1 = *(Nu + l + 1);
        *dNudx_l = ck_Hp*l/(2.0*l + 1.0)*Nu_lmin1 - ck_Hp*(l + 1.0)/(2.0*l + 1.0)*Nu_lplus1;
      }
    }
  }

  else
  {
    Psi = -Phi - 12.0*H0*H0/(c*c*k*k*a*a)*(Omega_gamma0*Theta2);
    dPhidx = Psi - ck_Hp*ck_Hp/3.0*Phi + H0*H0/(2.0*Hp*Hp)*(OmegaCDM0/a*delta_cdm \
                                                              + OmegaB0/a*delta_b \
                                                              + 4*Omega_gamma0/(a*a)*Theta0);
  }

  ddelta_cdmdx = ck_Hp*v_cdm - 3*dPhidx;
  ddelta_bdx   = ck_Hp*v_b - 3*dPhidx;

  dv_cdmdx     = -v_cdm - ck_Hp*Psi;
  dv_bdx       = -v_b - ck_Hp*Psi + dtaudx*R*(3*Theta1 + v_b);

  double *dTheta0dx = dThetadx;
  double *dTheta1dx = dThetadx + 1;

  // Derivatives of monopole and dipole
  *dTheta0dx = -ck_Hp*Theta1 - dPhidx;
  *dTheta1dx = ck_Hp/3.0*Theta0 - 2.0*ck_Hp/3.0*Theta2 + ck_Hp/3.0*Psi + dtaudx*(Theta1 + 1.0/3.0*v_b);

  // Derivatives of higher order multipoles
  for (int l = 2; l < Constants.n_ell_theta; ++l)
  {
    double *dThetaldx         = dThetadx + l;
    const double Thetal      = *(Theta + l);
    const double Thetalmin1  = *(Theta + l - 1);

    // Special formula for l_max
    if (l == Constants.n_ell_theta - 1)
    {
      *dThetaldx = ck_Hp*Thetalmin1 - c/(Hp*eta)*(l + 1.0)*Thetal + dtaudx*Thetal;
    }
    else
    {
      const double Thetalplus1 = *(Theta + l + 1);

      *dThetaldx = l*ck_Hp/(2.0*l + 1.0)*Thetalmin1 - (l + 1.0)/(2.0*l + 1.0)*ck_Hp*Thetalplus1 + \
                 + dtaudx*Thetal;

      // Special formula for Theta_2
      if (l == 2)
      {
        *dThetaldx -= dtaudx/10.0*Pi;
      }

    }
  }

  if (polarization)
  {
    const double Theta_p0 = *Theta_p;
    const double Theta_p1 = *(Theta_p + 1);

    double *dTheta_pdx_0 = dTheta_pdx;
    *dTheta_pdx_0 = -ck_Hp*Theta_p1 + dtaudx*(Theta_p0 - 1.0/2.0*Pi);

    for (int l = 1; l < Constants.n_ell_thetap; ++l)
    {
      const double Theta_p_lmin1 = *(Theta_p + l - 1);
      const double Theta_p_l = *(Theta_p + l);

      double *dTheta_pdx_l = dTheta_pdx + l;

      if (l == Constants.n_ell_thetap - 1)
      {

        *dTheta_pdx_l = ck_Hp*Theta_p_lmin1 - c/(Hp*eta)*(l + 1.0)*Theta_p_l + dtaudx*Theta_p_l;
      }
      else
      {
        const double Theta_p_lplus1 = *(Theta_p + l + 1);

        *dTheta_pdx_l = l/(2.0*l + 1.0)*ck_Hp*Theta_p_lmin1 - (l + 1.0)/(2.0*l + 1.0)*ck_Hp*Theta_p_lplus1 \
                      + dtaudx*Theta_p_l;
      }

      if (l == 2)
      {
        *dTheta_pdx_l -= dtaudx/10.0*Pi;
      }
    }
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

double Perturbations::get_Psi(const double x, const double k) const{
  return Psi_spline(x,k);
}

double Perturbations::get_Source_T(const double x, const double k) const{
  return ST_spline(x,k);
}

double Perturbations::get_Theta(const double x, const double k, const int ell) const{
  return Theta_spline[ell](x,k);
}

double Perturbations::get_Theta_p(const double x, const double k, const int ell) const{
  return Theta_p_spline[ell](x,k);
}
double Perturbations::get_Nu(const double x, const double k, const int ell) const{
  return Nu_spline[ell](x,k);
}


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
    fp << get_Psi(x,k)       << " ";
    fp << "\n";
  };
  std::for_each(x_array.begin(), x_array.end(), print_data);
}
