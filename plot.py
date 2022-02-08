import numpy as np
import matplotlib.pyplot as plt

Mpc = 3.08567758e22
Gyr = 1e9*3600*24*365
c = 3e8

class BackgroundCosmology:
    def __init__(self, data_file):
        self.data = np.loadtxt(data_file)
        self.x, self.eta, self.Hp, self.t, self.dHpdx, self.ddHpddx = np.transpose(self.data[:, :6])

        self.OmegaB, self.OmegaCDM, self.OmegaLambda, self.OmegaR, self.OmegaNu, self.OmegaK = np.transpose(self.data[:, 6:])

        self.Omega_m = self.OmegaB + self.OmegaCDM
        self.Omega_r = self.OmegaR + self.OmegaNu

        self.today_index = np.argmin(np.abs(self.x))

        self.z_m_r_eq = self.Omega_m[self.today_index]/self.Omega_r[self.today_index] - 1
        self.z_m_lambda_eq = (self.OmegaLambda[self.today_index]/self.Omega_m[self.today_index])**(1/3) - 1
        self.z_acc_begin = 1/(self.Omega_m[self.today_index]/(2*self.OmegaLambda[self.today_index]))**(1/3) - 1

        self.x_m_r_eq = -np.log(1 + self.z_m_r_eq)
        self.x_m_lambda_eq = -np.log(1 + self.z_m_lambda_eq)
        self.x_acc_begin = -np.log(1 + self.z_acc_begin)

        m_eq_index = np.argmin(np.abs(self.x - self.x_m_r_eq))
        m_lambda_index = np.argmin(np.abs(self.x - self.x_m_lambda_eq))
        acc_begin_index = np.argmin(np.abs(self.x - self.x_acc_begin))

        self.t_m_r_eq = self.t[m_eq_index]/Gyr
        self.t_m_lambda_eq = self.t[m_lambda_index]/Gyr
        self.t_acc_begin = self.t[acc_begin_index]/Gyr

    def printInfo(self):
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

    def plot(self, quantities, imageName, ylabel, xlabel = r"$x$", legends = [], logscale = False, xlims = [], ylims = []):
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
            #ax.plot(self.x, self.t[0]/Gyr*np.exp(2*(self.x - self.x[0])), "k--")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if logscale:
            ax.set_yscale("log")

        if len(xlims) > 0:
            ax.set_xlim(xlims[0], xlims[1])

        if len(ylims) > 0:
            ax.set_ylim(ylims[0], ylims[1])


        if len(ylims) == 0:
            ymin = np.min(quantities)
            ymax = np.max(quantities)
        else:
            # Just making sure ymin and ymax are not outside the ylims (when these ylims are defined)
            ymin = np.min(quantities) if np.min(quantities) < ylims[0] else ylims[0]
            ymax = np.max(quantities) if np.max(quantities) < ylims[1] else ylims[1]

        vlines_x = [self.x_m_r_eq, self.x_m_lambda_eq, self.x_acc_begin]
        vline_text = ["Matter-rad. eq.", r"Matter-$\Lambda$ eq.", "Acc. starts"]

        ax.vlines(vlines_x, ymin = ymin, ymax = ymax, colors = ["k", "k", "k"], linestyles = "dashed")


        for i in range(len(vlines_x)):
            ax.text(vlines_x[i], ymax*1.05, vline_text[i], horizontalalignment = "center")

        fig.savefig("images/{}.pdf".format(imageName), bbox_inches = "tight")

cosmo = BackgroundCosmology("cosmology.txt")
cosmo.printInfo()

# Plotting the omegas against x
cosmo.plot([cosmo.Omega_r, cosmo.Omega_m, cosmo.OmegaLambda], "omegas", r"$\Omega_i(x)$",
            legends = [r"$\Omega_r$", r"$\Omega_m$", r"$\Omega_\Lambda$"])

# Plotting H(x)
cosmo.plot(cosmo.Hp/np.exp(cosmo.x)/(100e3/Mpc), "H(x)", r"$H(x) \; \left(\frac{100\mathrm{ km/s}}{Mpc}\right)$",
           logscale = True, xlims = [-12, 0], ylims = [1e-1, 1e8])

# Plotting Hp(x)
cosmo.plot(cosmo.Hp/(100e3/Mpc), "Hp(x)", r"$\mathcal{H}(x)$ ($\frac{100 \mathrm{ km/s}}{Mpc}}$)", logscale = True,
           xlims = [-12, 0], ylims = [1e-1, 1e3])

# PLotting 1/Hp dHp/dx
cosmo.plot(cosmo.dHpdx/cosmo.Hp, "Hp-1 dHpdx", r"$\frac{1}{\mathcal{H}} \frac{d\mathcal{H}}{dx}$")

# Plotting 1/Hp d^2 Hp/dx^2
cosmo.plot(cosmo.ddHpddx/cosmo.Hp, "Hp-1 ddHpddx", r"$\frac{1}{\mathcal{H}} \frac{d^2 \mathcal{H}}{dx^2}$")

# Plotting eta(x)
cosmo.plot(cosmo.eta/Mpc, "eta(x)", r"$\eta(x)$ (Mpc)", logscale = True, xlims = [-12, 0], ylims = [1, 1e5])

# Plotting eta(x)Hp(x)/c
cosmo.plot(cosmo.eta*cosmo.Hp/c, "eta(x)Hp(x)", r"$\frac{\eta(x)\mathcal{H}(x)}{c}$", xlims = [-15, 0], ylims = [0.75, 3])

# Plotting t(x)
cosmo.plot(cosmo.t/Gyr, "t(x)", r"$t(x)$ (Gyr)", logscale = True)

sn_data = np.loadtxt("sn_data.txt", skiprows = 1)

z_obs, d_L_obs, error = np.transpose(sn_data)

x_lum, d_L = np.transpose(np.loadtxt("lum_dist.txt"))

z = np.exp(-x_lum) - 1

fig, ax = plt.subplots()

ax.plot(z, d_L/(1e3*Mpc), "b-", label = r"Computed $d_L(z)$")
ax.errorbar(z_obs, d_L_obs, error, fmt = "o", label = r"Observed $d_L(z)$", color = "r", capsize = 5, markersize = 5)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$d_L$ (Gpc)")
ax.legend()

fig.savefig("images/Supernova distances.pdf")
