import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, interpolate
import healpy as hp

txt_prefix   = "txtfiles/"
image_prefix = "images/"

Mpc = 3.08567758e22 # Mpc in m
Gyr = 1e9*3600*24*365 # Gyr in s
c = 3e8 # Speed of light in m/s
m_H = 1.67e-27  # Mass of hydrogen atom in kg
m_e = 9.11e-31  # Mass of electron in kg
k_B = 1.38e-23  # Boltzmann's constant
hbar = 1.0545718e-34 # Reduced Planck constant
G = 6.67e-11 # Newton's gravitational constant
eV = 1.6e-19

H0 = 2.2e-18 # Hubble parameter today
h  = H0/(100e3*Mpc**(-1)) # Dimensionless Hubble parameter today
TCMB0 = 2.7255 # Temperature of CMB today
omega_B0 = 0.05 # Density parameter of baryons today
rho_crit_0 = 3*H0**2/(8*np.pi*G) # Critical density today

epsilon_0 = 13.6*eV

# Fontsize in the plots
fontsize = 15

def x_to_z(x):
    """" Returns the redshift corresponding to a given x """
    return np.exp(-x) - 1

def z_to_x(z):
    """ Returns the value of x corresponding to a given redshift z """
    return -np.log(1 + z)

class Plotter:
    """
    A plotter class which the other classes can inherit from. It only takes an array of x-values as input.
    The plotting of quantities as functions of x is then handled by this class.
    """

    def __init__(self, x):
        self.x = np.array(x)

    def plot(self, quantities, imageName, ylabel, xlabel = r"$x$", legends = [], logscale = False,
                   xlims = [], ylims = [], grid = False, savefig = True, dashed = [], colors = []):
        """
        A general function used for plotting. Can be supplied with given quantities and
        plots the quantities against x.

        - imageName is the name of the image-file (it is placed in the images/ directory, so the full
          file name is images/{imageName}.pdf. Don't need to supply the "images/" or ".pdf")

        - ylabel and xlabel are the x- and y-labels to put in the plot
        - legends is a list specifying the legends to give to each quantity plotted
        - logscale specifies whether to use a logarithmic y-axis or not.
        - providing xlims and ylims means the plot will be restricted to the given limits.
        - grid specifies whether to include a grid or not.
        """

        fig, ax = plt.subplots()

        if len(colors) == 0:
            colors = ["b", "g", "r", "c"]

        # ymin and ymax will just be the minimum and maximum y-values to include in the plot

        if len(ylims) == 0:
            ymin = np.min(quantities)
            ymax = np.max(quantities)
        else:
            # Just making sure ymin and ymax are not outside the ylims (when these ylims are defined)
            ymin = np.min(quantities) if np.min(quantities) < ylims[0] else ylims[0]
            ymax = np.max(quantities) if np.max(quantities) < ylims[1] else ylims[1]

        # Plotting vertical lines
        linestyles = ["dashed", "dotted", "dashdot"]
        for i in range(len(self.vlines_x)):
            x = self.vlines_x[i]
            y_range = np.linspace(ymin, ymax, 2)
            ax.plot([x, x], y_range, color = "k", alpha = 0.8, label = self.vline_text[i], linestyle = linestyles[i])

        # If quantities is multidimensional then we are plotting multiple
        # quantities against self.x
        if isinstance(quantities[0], np.ndarray):
            for i in range(len(quantities)):
                if len(dashed) > 0:
                    linestyle = "--" if dashed[i] else "-"
                else:
                    linestyle = "-"

                ax.plot(self.x, quantities[i], color = colors[i], linestyle = linestyle, label = legends[i])

        else:
            if len(legends) > 0:
                ax.plot(self.x, quantities, colors[0], label = legends[0])
            else:
                ax.plot(self.x, quantities, colors[0])

        if len(legends) > 0:
            ax.legend(fontsize = fontsize)

        ax.set_xlabel(xlabel, fontsize = fontsize)
        ax.set_ylabel(ylabel, fontsize = fontsize)
        ax.tick_params(axis = "both", labelsize = fontsize)

        if grid:
            ax.grid()

        if logscale:
            ax.set_yscale("log")

        if len(xlims) > 0:
            ax.set_xlim(xlims[0], xlims[1])

        if len(ylims) > 0:
            ax.set_ylim(ylims[0], ylims[1])

        if savefig:
            fig.savefig(image_prefix + "{}.pdf".format(imageName), bbox_inches = "tight")
        else:
            return fig, ax


class BackgroundCosmology(Plotter):
    def __init__(self, data_file):
        self.data = np.loadtxt(txt_prefix + data_file)
        x, self.eta, self.Hp, self.t, self.dHpdx, self.ddHpddx = np.transpose(self.data[:, :6])

        Plotter.__init__(self, x)

        # Reading in all the density parameters from the data-file
        self.OmegaB, self.OmegaCDM, self.OmegaLambda, self.OmegaR, self.OmegaNu, self.OmegaK = np.transpose(self.data[:, 6:])

        self.Omega_m = self.OmegaB + self.OmegaCDM
        self.Omega_r = self.OmegaR + self.OmegaNu

        # Index corresponding to today, so self.t[today_index] gives the age of the universe today etc.
        self.today_index = np.argmin(np.abs(self.x))

        # Redshift of matter-rad. eq., matter-lambda eq., and beinning of accelerated expansion
        self.z_m_r_eq = self.Omega_m[self.today_index]/self.Omega_r[self.today_index] - 1
        self.z_m_lambda_eq = (self.OmegaLambda[self.today_index]/self.Omega_m[self.today_index])**(1/3) - 1
        self.z_acc_begin = (2*self.OmegaLambda[self.today_index]/self.Omega_m[self.today_index])**(1/3) - 1

        # The x-values of the same three points in time as above
        self.x_m_r_eq = z_to_x(self.z_m_r_eq)
        self.x_m_lambda_eq = z_to_x(self.z_m_lambda_eq)
        self.x_acc_begin = z_to_x(self.z_acc_begin)

        self.t_m_r_eq = self.x_to_t(self.x_m_r_eq)/Gyr
        self.t_m_lambda_eq = self.x_to_t(self.x_m_lambda_eq)/Gyr
        self.t_acc_begin = self.x_to_t(self.x_acc_begin)/Gyr

        self.vlines_x = [self.x_m_r_eq, self.x_m_lambda_eq, self.x_acc_begin]
        self.vline_text = ["Matter-rad. eq.", r"Matter-$\Lambda$ eq.", "Acc. starts"]

    def x_to_t(self, x):
        """
        Computes the time corresponding to a given x
        """
        closest_index = np.argmin(np.abs(self.x - x))
        return self.t[closest_index]

class Recombination(Plotter):
    def __init__(self, data_file, cosmo):
        self.data = np.loadtxt(txt_prefix + data_file)
        self.cosmo = cosmo
        x, self.Xe, self.ne, self.tau, self.dtaudx, self.ddtauddx, self.g, self.dgdx, self.ddgddx = np.transpose(self.data)

        Plotter.__init__(self, x)

        tau_func = interpolate.interp1d(self.x, self.tau - 1)
        self.x_decoupling = optimize.root_scalar(tau_func, bracket = [-7.5, -6.5]).root
        self.z_decoupling = x_to_z(self.x_decoupling)
        self.t_decoupling = self.cosmo.x_to_t(self.x_decoupling)/Gyr

        Xe_func = interpolate.interp1d(self.x, self.Xe - 0.5)
        self.x_recomb = optimize.root_scalar(Xe_func, bracket = [-7.5, -6.5]).root
        self.z_recomb = x_to_z(self.x_recomb)
        self.t_recomb = self.cosmo.x_to_t(self.x_recomb)/Gyr

        self.vlines_x = [self.x_decoupling, self.x_recomb]
        self.vline_text = ["Decoup.", "Recomb."]

def printEvents(event_names, z, x, t):
    print("{:^17}       |\t{:^5} \t\t |\t       {:^8} \t| {:^5}".format("Event", "z", "x", "t [Gyr]"))
    print("------------------------|------------------------|------------------------------|-------------")

    indices = np.arange(len(z))

    # Sorting the events, in order of decreasing redshift
    sorted_indices = np.argsort(z)[::-1]

    for i in range(len(x)):
        event_name = event_names[sorted_indices[i]]
        z_i = z[sorted_indices[i]]
        x_i = x[sorted_indices[i]]
        t_i = t[sorted_indices[i]]

        # Don't want to print with decimals when z_i > 1
        if z_i > 1:
            print("{:^17}       |\t{:^5.0f} \t\t |\t       {:8^.3f} \t\t| {:5^.3e}".format(event_name, z_i, x_i, t_i))
        else:
            print("{:^17}       |\t{:^5.2f} \t\t |\t       {:8^.3f} \t\t| {:5^.3e}".format(event_name, z_i, x_i, t_i))

    print("\n")

cosmo = BackgroundCosmology("cosmology.txt")

"""
# Plotting the omegas against x
cosmo.plot([cosmo.Omega_r, cosmo.Omega_m, cosmo.OmegaLambda], "omegas", r"$\Omega_i(x)$",
            legends = [r"$\Omega_r$", r"$\Omega_m$", r"$\Omega_\Lambda$"],
            xlims = [-15, 5], ylims = [-0.02, 1.1])

# Plotting H(x)
cosmo.plot(cosmo.Hp/np.exp(cosmo.x)/(100e3/Mpc), "H(x)", r"$H(x) \; \left(\frac{100\mathrm{ km/s}}{Mpc}\right)$",
           logscale = True, xlims = [-12, 5], ylims = [1e-1, 1e8])

# Plotting Hp(x)
cosmo.plot(cosmo.Hp/(100e3/Mpc), "Hp(x)", r"$\mathcal{H}(x)$ ($\frac{100 \mathrm{ km/s}}{Mpc}}$)", logscale = True,
           xlims = [-12, 5], ylims = [1e-1, 1e3])

# PLotting 1/Hp dHp/dx
cosmo.plot(cosmo.dHpdx/cosmo.Hp, "Hp-1 dHpdx", r"$\frac{1}{\mathcal{H}} \frac{d\mathcal{H}}{dx}$", grid = True,
            xlims = [-15, 5], ylims = [-1.2, 1.2])

# Plotting 1/Hp d^2 Hp/dx^2
cosmo.plot(cosmo.ddHpddx/cosmo.Hp, "Hp-1 ddHpddx", r"$\frac{1}{\mathcal{H}} \frac{d^2 \mathcal{H}}{dx^2}$", grid = True,
            xlims = [-15, 5], ylims = [0, 1.5])

# Plotting eta(x)
cosmo.plot(cosmo.eta/Mpc, "eta(x)", r"$\eta(x)$ (Mpc)", logscale = True, xlims = [-12, 5], ylims = [1, 5e4])

# Plotting eta(x)Hp(x)/c
cosmo.plot(cosmo.eta*cosmo.Hp/c, "eta(x)Hp(x)", r"$\frac{\eta(x)\mathcal{H}(x)}{c}$", xlims = [-15, 0], ylims = [0.75, 3])

# Plotting t(x)
cosmo.plot(cosmo.t/Gyr, "t(x)", r"$t(x)$ (Gyr)", logscale = True, xlims = [-15, 5], ylims = [1e-10, 5e2])
"""

"""
Reading in and plotting the supernova-data
"""
"""
sn_data = np.loadtxt("sn_data.txt", skiprows = 1)

# Redshifts and lum. distances from data
z_obs, d_L_obs, error = np.transpose(sn_data)

# Computed lum. distances for the array x_lum containing x-values
x_lum, d_L = np.transpose(np.loadtxt("lum_dist.txt"))

# Redshifts for the given x-values
z = np.exp(-x_lum) - 1

fig, ax = plt.subplots()

ax.plot(z, d_L/(1e3*Mpc), "b-", label = r"Computed $d_L(z)$")
ax.errorbar(z_obs, d_L_obs, error, fmt = "o", label = r"Observed $d_L(z)$", color = "r", capsize = 5, markersize = 5)
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$z$", fontsize = fontsize)
ax.set_ylabel(r"$d_L$ (Gpc)", fontsize = fontsize)
ax.legend(fontsize = fontsize)
ax.tick_params(axis = "both", labelsize = fontsize)

fig.savefig(image_prefix + "Supernova distances.pdf", bbox_inches = "tight")

"""

rec = Recombination("recombination.txt", cosmo)
rec_no_He = Recombination("recombination_without_He.txt", cosmo)


fig, ax = rec.plot(rec.Xe, "Xe(x)", r"$X_e$", xlims = [-12, 0], grid = True, savefig = False)
ax.plot(rec_no_He.x, rec_no_He.Xe, "r--")

ax.legend(["Decoup.", "Recomb.", "With He", "Without He"])
ax.set_xlim(-10, -3)
ax.set_yscale("log")

fig.savefig("images/Xe(x).pdf", bbox_inches = "tight")

"""
rec.plot([rec.tau, -rec.dtaudx, rec.ddtauddx], "tau(x)", r"$\tau$", logscale = True,
          legends = [r"$\tau(x)$", r"$-\tau'(x)$", r"$\tau''(x)$"], xlims = [-12, 0], ylims = [1e-8, 1e7])
"""

rec.plot([rec.g, rec.dgdx/10, rec.ddgddx/200], "g(x)", r"$\tilde{g}$", legends = [r"$\tilde{g}$", r"$\tilde{g}'$"
                                      , r"$\tilde{g}''$"], xlims = [-7.5, -6.2], dashed = [False, True, True])

"""
rec.plot([rec.g, rec.dgdx/10, rec.ddgddx/200], "g(x)_zoomed_out", r"$\tilde{g}$", legends = [r"$\tilde{g}$", r"$\tilde{g}'$"
                                      , r"$\tilde{g}''$"], xlims = [-12, 0], dashed = [False, True, True])
"""

# Defining arrays containing the names, redshifts etc for the different events of interest
event_names = ["Matter-rad. eq.", "Matter-Lambda eq.", "Acc. begin", "Decoupling", "Recombination"]
z_events = [cosmo.z_m_r_eq, cosmo.z_m_lambda_eq, cosmo.z_acc_begin, rec.z_decoupling, rec.z_recomb]
x_events = [cosmo.x_m_r_eq, cosmo.x_m_lambda_eq, cosmo.x_acc_begin, rec.x_decoupling, rec.x_recomb]
t_events = [cosmo.t_m_r_eq, cosmo.t_m_lambda_eq, cosmo.t_acc_begin, rec.t_decoupling, rec.t_recomb]

printEvents(event_names, z_events, x_events, t_events)

z_no_He = [rec_no_He.z_decoupling, rec_no_He.z_recomb]
x_no_He = [rec_no_He.x_decoupling, rec_no_He.x_recomb]
t_no_He = [rec_no_He.t_decoupling, rec_no_He.t_recomb]

printEvents(event_names[3:], z_no_He, x_no_He, t_no_He)

def saha_fixed_Xe(x):
    """
    Takes the LHS of the Saha eq. minus the RHS for a given x = log(a).
    Assumes Xe = 0.5, and is used to find the x for which the Saha eq. predicts
    Xe = 0.5.
    """
    C = m_H/(omega_B0*rho_crit_0)*(m_e*k_B*TCMB0/(2*np.pi*hbar**2))**(3/2)
    b = epsilon_0/(k_B*TCMB0)

    LHS = 0.5

    return C*np.exp(x)**(3/2)*np.exp(-b*np.exp(x)) - LHS

"""
# Finds solution to Saha eq. for Xe = 0.5
root = optimize.root_scalar(saha_fixed_Xe, x0 = -7.2, x1 = -7.25).root
print("Saha approx.: Recombination at {:.2f}".format(root))

print("Freeze-out abundance: Xe(x = 0) = {:.2e}".format(rec.Xe[-1]))

recomb_temp = TCMB0/np.exp(rec.x_recomb)
print("Recombination temperature: {:.2f} eV".format(recomb_temp*k_B/eV))

fig, ax = plt.subplots()
ax.plot(cosmo.t/Gyr, np.exp(cosmo.x)*cosmo.eta/(1e3*Mpc))
fig.savefig(image_prefix + "test.pdf")
"""

class Perturbations(Plotter):
    def __init__(self, data_file, rec, cosmo):
        self.data = np.loadtxt(txt_prefix + data_file)
        x, self.delta_cdm, self.delta_b, self.v_cdm, self.v_b, \
           self.theta0, self.theta1, self.theta2, self.Phi, self.Psi = np.transpose(self.data)

        self.rec = rec
        self.cosmo = cosmo

        Plotter.__init__(self, x)

        self.k = float(data_file.replace("perturbations_k", "").replace(".txt", ""))
        self.legend = r"$k = {}$/Mpc".format(self.k)

        self.vlines_x = [self.rec.x_decoupling, self.cosmo.x_m_r_eq]
        self.vline_text = ["Decoupling", "Matter-rad. eq."]

"""
data = np.loadtxt("perturbations_k0.100000.txt")
x = data[:, 0]
delta_cdm = data[:, 1]
delta_b = data[:, 2]

theta0 = data[:, 5]
phi = data[:, -1]

fig, ax = plt.subplots()
#ax.plot(x, phi, "b-")
ax.plot(x, theta0, "b-")
#ax.plot(x, delta_cdm, "b-")
#ax.plot(x, delta_b, "b--")
#ax.set_yscale("log")
fig.savefig(image_prefix + "pert_k0.01.pdf", bbox_inches = "tight")
"""

pert1 = Perturbations("perturbations_k0.300000.txt", rec, cosmo)
pert2 = Perturbations("perturbations_k0.050000.txt", rec, cosmo)
pert3 = Perturbations("perturbations_k0.000100.txt", rec, cosmo)

pert1.plot([pert1.delta_cdm, pert2.delta_cdm, pert3.delta_cdm,
            np.abs(pert1.delta_b), np.abs(pert2.delta_b), np.abs(pert3.delta_b)],
           "pert_deltas", r"$\delta_\mathrm{CDM}, \delta_b$",
           legends = [pert1.legend, pert2.legend, pert3.legend, "", "", ""],
           logscale = True, dashed = [False, False, False, True, True, True],
           colors = ["b", "r", "g", "b", "r", "g"])

pert1.plot([pert1.v_cdm, pert2.v_cdm, pert3.v_cdm,
            np.abs(pert1.v_b), np.abs(pert2.v_b), np.abs(pert3.v_b)],
           "pert_v", r"$v_\mathrm{CDM}, v_b$",
           legends = [pert1.legend, pert2.legend, pert3.legend, "", "", ""],
           logscale = True, dashed = [False, False, False, True, True, True],
           colors = ["b", "r", "g", "b", "r", "g"])

pert1.plot([pert1.Phi, pert2.Phi, pert3.Phi], "pert_phi", r"$\mathrm{\Phi}$",
            legends = [pert1.legend, pert2.legend, pert3.legend],
            colors = ["b", "r", "g", "b", "r", "g"])

pert1.plot([pert1.Psi, pert2.Psi, pert3.Psi], "pert_psi", r"$\mathrm{\Psi}$",
            legends = [pert1.legend, pert2.legend, pert3.legend],
            colors = ["b", "r", "g", "b", "r", "g"])

pert1.plot([pert1.theta0, pert2.theta0, pert3.theta0], "pert_theta0", r"$\Theta_0$",
           legends = [pert1.legend, pert2.legend, pert3.legend],
           colors = ["b", "r", "g", "b", "r", "g"])

pert1.plot([pert1.theta1, pert2.theta1, pert3.theta1], "pert_theta1", r"$\Theta_1$",
           legends = [pert1.legend, pert2.legend, pert3.legend],
           colors = ["b", "r", "g", "b", "r", "g"])

def plot_power_spectra(filenames, labels, colors, outputname, xlabel = r"Multipole $\ell$", \
                       ylabel = r"$\ell \left(\ell + 1\right) C_\ell/2\pi$ ($\mu$K)$^2$"):
    fig, ax = plt.subplots()
    for i in range(len(filenames)):
        power_spect = np.loadtxt(txt_prefix + filenames[i])
        ell, Cell = np.transpose(power_spect)

        ax.plot(ell, Cell, colors[i], label = labels[i])

    ax.set_xscale("log")
    #ax.set_xlabel(r"Multipole $\ell$")
    #ax.set_ylabel(r"$\ell \left(\ell + 1\right) C_\ell/2\pi$ ($\mu$K)$^2$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(outputname, bbox_inches = "tight")


"""
Plotting matter power spectrum
"""

matter_power = np.loadtxt(txt_prefix + "matter_power.txt")
k, P_k = np.transpose(matter_power)

x = k*(Mpc/h)
y = P_k*(h/Mpc)**3


fig, ax = plt.subplots()
ax.plot(x, y, "r-", label = "Theory prediction")

# Reading in and plotting data of matter power spectrum from observations
matter_power_obs = np.loadtxt(txt_prefix + "matter_power_spect_data.txt", skiprows = 0)
k_obs, P_k_obs, errors = np.transpose(matter_power_obs)

ax.errorbar(k_obs, P_k_obs, errors, fmt = "o", label = "SDSS DR7 LRG",
            color = "k", capsize = 2, markersize = 2)

# x and Hp at matter-radiation equality
x_eq = cosmo.x_m_r_eq
Hp_eq = np.interp(x_eq, cosmo.x, cosmo.Hp)

# The equality scale (in h/Mpc)
k_eq = Hp_eq/c*(Mpc/h)

y_range = [np.min(y), np.max(y)]

ax.plot([k_eq, k_eq], y_range, "k--", alpha = 0.8, label = r"$k_\mathrm{eq}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel(r"$k$ ($h$/Mpc)")
ax.set_ylabel(r"$P(k)$ (Mpc/$h$)$^3$")
fig.savefig(image_prefix + "matter_power_spectrum.pdf", bbox_inches = "tight")

"""
Plotting CMB power spectrum
"""

power_spectrum = np.loadtxt(txt_prefix + "cells.txt")
ells, C_ells = np.transpose(power_spectrum)

fig, ax = plt.subplots()

ax.plot(ells, C_ells, "r-", label = "Theory prediction")

#  Reading in and plotting the data from CMB observations
cmb_data = np.loadtxt(txt_prefix + "cmb_data.txt", skiprows = 0)
x_obs, y_obs = cmb_data[:, 0], cmb_data[:, 1]
errors = np.array([i[::-1] for i in cmb_data[:, 2:]]).T

ax.errorbar(x_obs, y_obs, errors, fmt = "o", label = "Planck 2018",
            color = "k", capsize = 2, markersize = 2)

ax.set_xscale("log")
ax.set_xlabel(r"Multipole $\ell$")
ax.set_ylabel(r"$\ell \left(\ell + 1\right) C_\ell/2\pi$ ($\mu$K)$^2$")
ax.legend()
fig.savefig(image_prefix + "cells.pdf", bbox_inches = "tight")

np.random.seed(2492938)

def plot_cmb_map(C_ells):
    """
    Plots the CMB map with the Mollweide projection, given an array of C_ells
    """
    nside = 128
    dpi = 500

    from matplotlib.colors import ListedColormap
    import numpy as np
    cmap = ListedColormap(np.loadtxt(txt_prefix + "Planck_Parchment_RGB.txt")/255.)
    cmap.set_bad("gray")
    cmap.set_under("white")

    fig, ax = plt.subplots()
    m = hp.sphtfunc.synfast(C_ells, nside)
    hp.mollview(m, fig, title = "", cbar = False, cmap = cmap, remove_dip = True)

    ax.axis("off")
    fig.savefig(image_prefix + "cmb_map.pdf", bbox_inches = "tight", dpi = dpi)

# Need to remove the multiplying factor that is used when plotting the power spectrum
C_ells /= ells*(ells + 1)/(2*np.pi)

# Adding two more values in the beginning of C_ells_full, corresponding to monopole and dipole
# We don't need them, we remove them in hp.mollview, but healpy expects them to be in the C_ells array (I think?)
C_ells_full = np.zeros(len(C_ells) + 2)
C_ells_full[2:] = C_ells

plot_cmb_map(C_ells_full)

"""
Plotting power spectra for varying cosmological parameters
"""

filenames = ["cells.txt", "cells_lambda0.9.txt", "cells_m_0.5.txt", "cells_b_0.1.txt"]
labels = ["Fiducial model", r"$\Omega_{\Lambda0} = 0.9$",
         r"$\Omega_{\mathrm{M}0} = 0.5$", r"$\Omega_{\mathrm{B}0} = 0.1$"]
colors = ["b-", "r-", "g--", "m:"]
outputname = image_prefix + "cells_varying_params.pdf"

plot_power_spectra(filenames, labels, colors, outputname)

"""
Plotting power spectra when only including the first, second, third or fourth term
in the source function
"""

filenames = ["cells_first_term.txt", "cells_second_term.txt",
             "cells_third_term.txt", "cells_fourth_term.txt"]

#labels = ["1st term", "2nd term", "3rd term", "4th term"]

labels = ["SW", "ISW", "Doppler", "Fourth term"]
colors = ["b-", "r-", "g--", "m:"]
outputname = image_prefix + "cells_source_terms.pdf"

plot_power_spectra(filenames, labels, colors, outputname)

ells = [10, 100, 1000]
filenames = [txt_prefix + "theta{}.txt".format(i) for i in ells]
colors = ["b-", "g-", "r-"]
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range(len(filenames)):
    theta_l = np.loadtxt(filenames[i])
    k, theta = np.transpose(theta_l)

    ax1.plot(c*k/H0, theta, colors[i], label = r"$\ell = {}$".format(ells[i]))
    ax2.plot(c*k/H0, theta**2/(c*k)*H0, colors[i], label = r"$\ell = {}$".format(ells[i]))

ax1.legend()
ax1.set_xlabel(r"$ck/H_0$")
ax1.set_ylabel(r"$\Theta_l$")
ax1.set_xscale("log")
fig1.savefig(image_prefix + "theta_l.pdf", bbox_inches = "tight")

ax2.legend()
ax2.set_xlabel(r"$ck/H_0$")
ax2.set_ylabel(r"$\frac{\Theta_l^2}{k} \cdot \left(\frac{H_0}{c}\right)$")
ax2.set_yscale("log")
ax2.set_ylim(1e-10, 2e-4)
ax2.set_xscale("log")
fig2.savefig(image_prefix + "C_ell_integrand.pdf", bbox_inches = "tight")

print(cosmo.eta[-1]/Mpc)
