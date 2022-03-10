#ifndef _BACKGROUNDCOSMOLOGY_HEADER
#define _BACKGROUNDCOSMOLOGY_HEADER
#include <iostream>
#include <fstream>
#include "Utils.h"

using Vector = std::vector<double>;

class BackgroundCosmology{
  private:

    // Cosmological parameters
    double h;                       // Little h = H0/(100km/s/Mpc)
    double OmegaB;                  // Baryon density today
    double OmegaCDM;                // CDM density today
    double OmegaLambda;             // Dark energy density today
    double Neff;                    // Effective number of relativistic species (3.046 or 0 if ignoring neutrinos)
    double TCMB;                    // Temperature of the CMB today in Kelvin

    // Derived parameters
    double OmegaR;                  // Photon density today (follows from TCMB)
    double OmegaNu;                 // Neutrino density today (follows from TCMB and Neff)
    double OmegaK;                  // Curvature density = 1 - OmegaM - OmegaR - OmegaNu - OmegaLambda
    double H0;                      // The Hubble parameter today H0 = 100h km/s/Mpc

    double rho_crit_0;              // The critical density of the universe today

    // Start and end of x-integration (can be changed)
    double x_start = Constants.x_start;
    double x_end = Constants.x_end;

    // Splines to be made
    Spline eta_of_x_spline{"eta"};
    Spline t_of_x_spline{"t"};

  public:

    // Constructors
    BackgroundCosmology() = delete;
    BackgroundCosmology(
        double h,
        double OmegaB,
        double OmegaCDM,
        double OmegaK,
        double Neff,
        double TCMB
        );

    // Print some useful info about the class
    void info() const;

    // Changing the x-range of integration from outside the class
    void set_xlims(double x_min, double x_max);

    // Do all the solving
    void solve();

    // Calculates the comoving distance to an object whose light (emitted when the universe had scale factor a = exp(x))
    // reached us today
    double comoving_distance(double x) const;

    // Calculates the comoving radial coordinate of an object at comoving distance given by the comiving_distance function
    double r_coordinate(double x) const;

    // Calculates the angular diameter distance for a given x
    double d_A(double x) const;

    // Calculates the luminosity distance for a given x
    double d_L(double x) const;

    // Returns the age of the universe for a particular x given as input
    double age_of_universe(double x) const;

    // Output some results to file (eta(x), H(x), Hp(x), etc.)
    void output(const std::string filename) const;

    // Get functions returning the value implied by the function name (eta_of_x -> eta(x) etc.)
    // All the functions take in some x and return the value of the quantity at that x
    // The expressions used in the derivative functions are derived in the appendices of the report
    double eta_of_x(double x) const;
    double t_of_x(double x) const;
    double H_of_x(double x) const;
    double Hp_of_x(double x) const;
    double dHdx_of_x(double x) const;
    double ddHddx_of_x(double x) const;
    double dHpdx_of_x(double x) const;
    double ddHpddx_of_x(double x) const;
    double get_OmegaB(double x = 0.0) const;
    double get_OmegaM(double x = 0.0) const;
    double get_OmegaR(double x = 0.0) const;
    double get_OmegaRtot(double x = 0.0) const;
    double get_OmegaNu(double x = 0.0) const;
    double get_OmegaCDM(double x = 0.0) const;
    double get_OmegaLambda(double x = 0.0) const;
    double get_OmegaK(double x = 0.0) const;
    double get_OmegaMnu(double x = 0.0) const;
    double get_H0() const;
    double get_h() const;
    double get_Neff() const;
    double get_TCMB(double x = 0.0) const;
    double get_n_b(double x) const;
};

#endif
