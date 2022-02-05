import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, imageName, colors, xlabel, ylabel, legends = [], logscale = False, xlims = [], ylims = []):
    fig, ax = plt.subplots()

    if np.array(x).ndim > 1:
        for i in range(len(x)):
            ax.plot(x[i], y[i], colors[i], label = legends[i])

        ax.legend()

    else:
        ax.plot(x, y, colors[0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logscale:
        ax.set_yscale("log")

    if len(xlims) > 0:
        ax.set_xlim(xlims[0], xlims[1])

    if len(ylims) > 0:
        ax.set_ylim(ylims[0], ylims[1])

    fig.savefig("images/{}.pdf".format(imageName))

Mpc = 3.08567758e22
c = 3e8

data = np.loadtxt("cosmology.txt")

x = data[:, 0]
eta = data[:, 1]
Hp = data[:, 2]

OmegaB, OmegaCDM, OmegaLambda, OmegaR, OmegaNu, OmegaK = np.transpose(data[:, 4:])

Omega_m = OmegaB + OmegaCDM
Omega_r = OmegaR + OmegaNu

# Plotting Hp(x)
plot(x, Hp/(100e3/Mpc), "Hp(x)", ["b-"], r"$x$", r"$\mathcal{H}(x)\; \left(\frac{100\mathrm{ km/s}}{Mpc}\right)$",
     logscale = True, xlims = [-12, 0], ylims = [1e-1, 1e3])

# Plotting H(x)
H = Hp/np.exp(x)
plot(x, H/(100e3/Mpc), "H(x)", ["b-"], r"$x$", r"$H(x)\; \left(\frac{100\mathrm{ km/s}}{Mpc}\right)$",
     logscale = True, xlims = [-12, 0], ylims = [1e-1, 1e8])

# Plotting eta(x)
plot(x, eta/Mpc, "eta(x)", ["b-"], r"$x$", r"$\eta(x)\; (Mpc)$", logscale = True, xlims = [-12, 0], ylims = [1, 1e5])

# Plotting eta(x)Hp(x)/c
plot(x, eta*Hp/c, "eta(x)Hp(x)", ["b-"], r"$x$", r"$\frac{\eta(x)\mathcal{H}(x)}{c}$",
     xlims = [-15, 0], ylims = [0.75, 3])

plot([x, x, x], [Omega_m, Omega_r, OmegaLambda], "omegas", ["b-", "r-", "g-"], r"$x$", r"$\Omega_i(x)$",
     [r"$\Omega_\mathrm{matter} = \Omega_b + \Omega_\mathrm{CDM}$",
      r"$\Omega_\mathrm{relativistic} = \Omega_r + \Omega_\nu$",
      r"$\Omega_\Lambda$"])
