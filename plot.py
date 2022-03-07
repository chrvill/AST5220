import numpy as np
import matplotlib.pyplot as plt

Mpc = 3.08567758e22 # Mpc in m
Gyr = 1e9*3600*24*365 # Gyr in s
c = 3e8 # Speed of light in m/s

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
                   xlims = [], ylims = [], grid = False):
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

        colors = ["b-", "g-", "r-"]

        # If quantities is mutlidimensional then we are plotting multiple
        # quantities against self.x
        if isinstance(quantities[0], np.ndarray):
            for i in range(len(quantities)):
                ax.plot(self.x, quantities[i], colors[i], label = legends[i])

            ax.legend()

        else:
            ax.plot(self.x, quantities, colors[0])

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

        # ymin and ymax will just be the minimum and maximum y-values to include in the plot

        if len(ylims) == 0:
            ymin = np.min(quantities)
            ymax = np.max(quantities)
        else:
            # Just making sure ymin and ymax are not outside the ylims (when these ylims are defined)
            ymin = np.min(quantities) if np.min(quantities) < ylims[0] else ylims[0]
            ymax = np.max(quantities) if np.max(quantities) < ylims[1] else ylims[1]

        # Will include dashed vertical lines. vlines_x specifies the x-coordinates of these lines,
        # while vline_text specifies the text belonging to each line.

        # Plotting the vertical lines
        ax.vlines(self.vlines_x, ymin = ymin, ymax = ymax, colors = ["k", "k", "k"], linestyles = "dashed")

        # x-coordinates of text are determined through trial and error, just making sure the text
        # doesn't overlap for "Matter-rad. eq." and "Acc. starts"
        ax.text(self.vlines_x[0], ymax*1.05, self.vline_text[0], horizontalalignment = "center", fontsize = fontsize)
        ax.text(self.vlines_x[1]*0.5, ymax*1.05, self.vline_text[1], horizontalalignment = "left", fontsize = fontsize)
        ax.text(self.vlines_x[2]*1.5, ymax*1.05, self.vline_text[2], horizontalalignment = "right", fontsize = fontsize)

        fig.savefig("images/{}.pdf".format(imageName), bbox_inches = "tight")


class BackgroundCosmology(Plotter):
    def __init__(self, data_file):
        self.data = np.loadtxt(data_file)
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
        #self.z_acc_begin = 1/(self.Omega_m[self.today_index]/(2*self.OmegaLambda[self.today_index]))**(1/3) - 1
        self.z_acc_begin = (2*self.OmegaLambda[self.today_index]/self.Omega_m[self.today_index])**(1/3) - 1

        # The x-values of the same three points in time as above
        self.x_m_r_eq = z_to_x(self.z_m_r_eq)
        self.x_m_lambda_eq = z_to_x(self.z_m_lambda_eq)
        self.x_acc_begin = z_to_x(self.z_acc_begin)

        # Indices for when the three aforementioned points in time occur
        m_eq_index = np.argmin(np.abs(self.x - self.x_m_r_eq))
        m_lambda_index = np.argmin(np.abs(self.x - self.x_m_lambda_eq))
        acc_begin_index = np.argmin(np.abs(self.x - self.x_acc_begin))

        # Times at which the aforementioned points in time occur
        self.t_m_r_eq = self.t[m_eq_index]/Gyr
        self.t_m_lambda_eq = self.t[m_lambda_index]/Gyr
        self.t_acc_begin = self.t[acc_begin_index]/Gyr

        self.vlines_x = [self.x_m_r_eq, self.x_m_lambda_eq, self.x_acc_begin]
        self.vline_text = ["Matter-rad. eq.", r"Matter-$\Lambda$ eq.", "Acc. starts"]

    def printInfo(self):
        """
        Prints the z-, x- and t-values corresponding to matter-rad. eq.,
        matter-lambda eq. and beginning of accelerated expansion.
        """

        print("Age of universe: {:.2f}".format(self.t[self.today_index]/Gyr))
        print("Eta today: {:.2f}".format(self.eta[self.today_index]/(c*Gyr)))

        print("\t\t |\t z \t\t | \t\t x \t\t|  t (Gyr)")
        print("-----------------|-----------------------|------------------------------|-----------------")
        print("Matter-rad. eq.  |      {:^.0f} \t\t |             {:^.3f} \t\t| {:^.3e}".format(self.z_m_r_eq, self.x_m_r_eq,
                                                                                                self.t_m_r_eq))
        print("Matter-Lambda eq |      {:^.2f} \t\t |             {:^.3f}  \t\t| {:^.3f}".format(self.z_m_lambda_eq,
                                                                                                 self.x_m_lambda_eq,
                                                                                                 self.t_m_lambda_eq))
        print("Acc. begins      |      {:^.2f} \t\t |             {:^.3f}  \t\t| {:^.3f}".format(self.z_acc_begin,
                                                                                                 self.x_acc_begin,
                                                                                                 self.t_acc_begin))

class Recombination(Plotter):
    def __init__(self, data_file):
        self.data = np.loadtxt(data_file)
        x, self.Xe, self.ne, self.tau, self.dtaudx, self.ddtauddx, self.g, self.dgdx, self.ddgddx = np.transpose(self.data)

        Plotter.__init__(self, x)

        self.decoupling_index = np.argmin(np.abs(self.tau - 1))
        self.x_decoupling = self.x[self.decoupling_index]
        self.z_decoupling = x_to_z(self.x_decoupling)

        self.recomb_index = np.argmin(np.abs(self.Xe - 0.5))
        self.x_recomb = self.x[self.recomb_index]
        self.z_recomb = x_to_z(self.x_recomb)

        print(self.z_decoupling)
        print(self.z_recomb)


        self.vlines_x = [-1, -1, -1]
        self.vline_text = ["", "", ""]

cosmo = BackgroundCosmology("cosmology.txt")
cosmo.printInfo()

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
Reading in and plotting the supernova-data
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

fig.savefig("images/Supernova distances.pdf", bbox_inches = "tight")

rec = Recombination("recombination.txt")
rec.plot(rec.Xe, "Xe(x)", r"$X_e$", xlims = [-12, 0])#, logscale = True)
rec.plot([rec.tau, -rec.dtaudx], "tau(x)", r"$\tau$", logscale = True, legends = [r"$\tau(x)$", r"$\tau'(x)$"],
          ylims = [1e-8, 1e7])

g = rec.g/np.max(rec.g)
dgdx = rec.dgdx/np.max(np.abs(rec.dgdx))
ddgddx = rec.ddgddx/np.max(np.abs(rec.ddgddx))
rec.plot([g, dgdx, ddgddx], "g(x)", r"$\tilde{g}$", legends = [r"$\tilde{g}$", r"$\tilde{g}'$"
                                      , r"$\tilde{g}''$"], xlims = [-9, 0])
