import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.integrate as integrate
from scipy.optimize import Bounds
from density import density as Density
from scipy.optimize import least_squares
import scipy.optimize as optim

import scienceplots

plt.style.use("science")
BETA = 0.325  # order-parameter exponent
ALPHA = 0.11  # heat-capacity exponent (for diameter correction)


def rho_z_integrate(z, A):
    rhol = A[0]
    rhog = A[1]
    z0 = A[2]
    d = A[3]
    rho = 0.5 * (rhol + rhog) - 0.5 * (rhol - rhog) * np.tanh((2 * (z - z0)) / d)
    return rho


def rho_z(A, z):
    rhol = A[0]
    rhog = A[1]
    z0 = A[2]
    d = A[3]
    rho = 0.5 * (rhol + rhog) - 0.5 * (rhol - rhog) * np.tanh((2 * (z - z0)) / d)
    return rho


def rho_z_cost(A, z, density):
    return density - rho_z(A, z)


def fit_rho_z(zbin, density, A0=np.array([800, 20, 60, 20])):  # rho_l, rho_g, z0, d

    res = optim.least_squares(
        rho_z_cost,
        A0,
        bounds=((0, 0, 0, 0), (2e3, 1e3, 100, 100)),
        args=(zbin, density),
        method="trf",  # bounded, robust LS solver
        loss="linear",
    )  # ordinary LS

    return res.x


def fit_density(
    df,
    bin_width=4,  # Angstroms
    time_avg=True,
    mode="molecule",
    std=True,
    absval=True,
    center=True,
    auto_range=False,
    norm="mass",
    actime=False,
    start=0,
):

    density, zbin, mass_err = Density(
        df,
        bin_width=bin_width,
        time_avg=time_avg,
        mode=mode,
        std=std,
        absval=absval,
        center=center,
        auto_range=auto_range,
        norm=norm,
        actime=actime,
        start=start,
    )

    zmin = zbin.min()
    zmax = zbin.max()
    A0 = np.array([800, 20, 60, 20])  # rho_l, rho_g, z0, d

    A = fit_rho_z(zbin, density, A0)
    rho_l, rho_g, z0, d = A
    rho = rho_z(A, zbin)

    total_system_mass = df["molecule"]["mass"] * df["nmolecule"] * df["AMU_TO_KG"]
    L = utils.get_L(df)
    V = (zbin[1] - zbin[0]) * L[0] * L[1] * 1e-10**3
    frac_density = density * V / total_system_mass  # TODO finish fractional density

    A0 = np.array([0.01, 0.0005, 60, 20])  # rho_l, rho_g, z0, d

    frac_A = fit_rho_z(zbin, frac_density, A0)
    frac_rho_l, frac_rho_g, frac_z0, frac_d = frac_A
    # frac_rho = rho_z(frac_A, zbin)

    frac_liq, err_liq = integrate.quad(
        rho_z_integrate, 0, frac_z0, args=(frac_A), full_output=0
    )
    frac_gas, err_gas = integrate.quad(
        rho_z_integrate, frac_z0, zmax, args=(frac_A), full_output=0
    )

    total = frac_gas + frac_liq
    frac_gas /= total
    frac_liq /= total

    return (
        rho,
        density,
        zbin,
        rho_l,
        rho_g,
        d,
        z0,
        mass_err,
        frac_liq,
        frac_gas,
    )


def plot_density_fit(
    rho,
    density,
    zbin,
    rho_l,
    rho_g,
    d,
    z0,
    mass_err,
    frac_liquid,
    frac_gas,
    frac_liquid_kmeans,
    frac_gas_kmeans,
    title="",
    figsize=(8, 6),
    extras=False,
    std=False,
):

    zmin = zbin.min()
    zmax = zbin.max()
    interface_start = z0 - d / 2
    interface_end = z0 + d / 2
    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    # ylim = (0, 1.1 * max(rho.max(), density.max()))

    ax.set_xlabel(r"$Z \ (\text{Å})$")
    ax.set_ylabel(r"$\rho \ (\mathrm{g cm^{-3}})$")
    # ax.set_title(f"{mol} {temp}K Density Profile")
    ax.set_title(title)
    ax.set_xlim(zmin, zmax)
    # ax.set_ylim(ylim)
    ax.plot(zbin, density, label="data", color="navy")
    ax.plot(zbin, rho, label="fit", color="goldenrod")
    ylim = ax.get_ylim()
    ax.vlines(
        z0,
        0,
        ylim[1],
        color="black",
        linestyle="--",
        label="z0",
        zorder=-1,
    )

    ax.hlines(
        rho_l,
        zmin,
        zmax,
        linestyle=(0, (5, 10)),
        color="black",
        zorder=0,
    )

    if std:
        ax.fill_between(
            zbin, density - mass_err, density + mass_err, color="navy", alpha=0.3
        )

    ax.text(zmin + 0.6 * zmax, rho_l * 1.03, r"$\rho_l$", fontsize=12)
    ax.text(zmax * 0.694, rho_g + 95, r"$\rho_g$", fontsize=12)

    if extras:
        interface_start_arg = np.argmin(np.abs(zbin - interface_start))
        interface_end_arg = np.argmin(np.abs(zbin - interface_end))
        interface_start_y = rho[interface_start_arg]
        interface_end_y = rho[interface_end_arg]
        ax.vlines(
            interface_start,
            0,
            interface_start_y,
            color="black",
            linestyle="--",
            label="d",
            zorder=0,
        )
        ax.vlines(
            interface_end,
            0,
            interface_end_y * 0.95,
            color="black",
            linestyle="--",
            zorder=0,
        )

        gas_start = np.argmin(np.abs(zbin - z0))

        ax.fill_between(
            zbin[gas_start:],
            rho[gas_start:],
            color="tab:red",
            alpha=0.3,
            label="gas",
        )

        ax.fill_between(
            zbin[: gas_start + 1],
            rho[: gas_start + 1],
            color="tab:blue",
            alpha=0.3,
            label="liq",
            zorder=0,
        )
        ax.text(15, 0.45 * ylim[1], f"{frac_liquid * 100:.1f}%", fontsize=12)
        ax.text(
            15,
            0.40 * ylim[1],
            f"{frac_liquid_kmeans * 100:.1f}%",
            fontsize=12,
            color="tab:green",
        )
        ax.text(
            0.62 * zmax,
            rho_g + 35,
            f"{frac_gas * 100:.1f}%",
            fontsize=12,
        )
        ax.text(
            0.73 * zmax,
            rho_g + 35,
            f"{frac_gas_kmeans * 100:.1f}%",
            fontsize=12,
            color="tab:green",
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig, ax


def init_B_rhoc(rl, rg, T, Tc):
    # y ≈ rho_c + B*(Tc_fixed - T) == a + m*T with m=-B, a=rho_c + B*Tc_fixed

    y = (rl + rg) / 2
    x = T
    m, b = np.polyfit(x, y, deg=1)
    B = m * -1
    rho_c = b - B * Tc
    return B, rho_c


def init_Tc_A(T, rl, rg):

    # get a Tc and A guess by matching the fitted slope to the known beta

    Tmax = T.max()
    Tc_grid = np.linspace(Tmax + 5.0, Tmax + 75.0, 5)
    best = None
    for Tc in Tc_grid:

        x = np.log10(Tc - T)
        y = np.log10((rl + rg) / 2)

        beta_fit, b = np.polyfit(x, y, deg=1)
        A = 10**b
        score = (beta_fit - BETA) ** 2
        if (best is None) or (score < best[0]):

            best = (score, Tc, A)

    _, Tc0, A0 = best
    return Tc0, A0


def residuals(p, rho_l, rho_g, T, corrected=False):

    if corrected:
        A, Tc, B, rhoc, C = p
    else:
        A, Tc, B, rhoc = p

    t = np.abs(Tc - T)
    # universal scaling law
    r1 = (rho_l - rho_g) - A * (Tc - T) ** BETA
    # law of rectilinear diameters
    r2 = (rho_l + rho_g) / 2 - (rhoc + B * (Tc - T))

    if corrected:

        r2 += C * t ** (1 - ALPHA)

    r1 /= np.std(rho_l - rho_g)
    r2 /= np.std((rho_l + rho_g) / 2)

    return np.hstack((r1, r2))


def fit_critical(
    rho_l,
    rho_g,
    T,
    corrected=False,
    p0=[100.0, 450.0, 1.0, 300.0, 0],  # A, Temp, B, Rho, C
    bounds=[
        [0, 400.0, 0, 100.0, -np.inf],
        [2000.0, 600.0, 1000, 600.0, np.inf],
    ],  # lower, upper
    C0=0,
):
    """
    Returns a dict with the fitted parameters and the final cost.
    """
    lb = bounds[0]
    ub = bounds[1]

    if not corrected and p0 and len(p0) == 5:
        p0 = p0[:-1]
    if not corrected and len(lb) == 5:
        lb = bounds[0][:-1]
        ub = bounds[1][:-1]

    if not p0:
        Tc0, A0 = init_Tc_A(T, rho_l, rho_g)
        B0, rhoc0 = init_B_rhoc(rho_l, rho_g, T, Tc0)
        if not C0:
            C0 = 0
        p0 = [A0, Tc0, B0, rhoc0]
        if corrected:
            p0.append(C0)

    res = least_squares(
        residuals,
        p0,
        bounds=[lb, ub],
        args=(rho_l, rho_g, T, corrected),
        method="trf",  # bounded, robust LS solver
        loss="linear",
    )  # ordinary LS

    if not corrected:
        A, Tc, B, rhoc = res.x
        ret = {"A": A, "Tc": Tc, "B": B, "rhoc": rhoc, "cost": res.cost}

    if corrected:
        A, Tc, B, rhoc, C = res.x
        ret = {"A": A, "Tc": Tc, "B": B, "rhoc": rhoc, "C": C, "cost": res.cost}
    return ret


def coexistence_densities(T, Tc, rhoc, A, B, beta):
    """
    Return liquid and vapour densities for an array-like T.

    Parameters
    ----------
    T      : array_like  – temperature(s) at which to evaluate (K)
    Tc     : float       – critical temperature (K)
    rhoc   : float       – critical density (same units as output)
    A, B   : floats      – fitting constants
    beta   : float       – critical exponent

    Returns
    -------
    rho_l, rho_v : ndarray
        Liquid and vapour densities at each T, same shape as T.
    """
    dT = Tc - np.asarray(T)  # works for scalars or arrays
    dp = A * dT**beta
    Σρ = 2 * (rhoc + B * dT)

    rho_l = (Σρ + dp) / 2
    rho_v = (Σρ - dp) / 2
    return rho_l, rho_v


def test_fit():
    T = np.array([250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0])
    rho_l = np.array([856.00, 822.37, 787.21, 748.26, 703.07, 648.97, 574.92])
    rho_g = np.array([-8.64e-02, 0.221, 1.488, 4.714, 12.197, 27.819, 63.131])

    return fit_critical(rho_l, rho_g, T)


######################################### Archive ##########################################

# def T_c(A, T):

#     a = A[0]
#     Tc = A[1]
#     beta = 0.325
#     diff = a * (Tc - T) ** beta
#     return diff


# def T_c_cost(A, rho_l, rho_g, T):
#     err = np.linalg.norm((rho_l - rho_g) - T_c(A, T))
#     return err


# def fit_T_c(
#     rho_ls,
#     rho_gs,
#     temps,
#     bounds=Bounds(lb=[0, 300], ub=[200, 500]),
#     x0=[100, 400],
#     maxiter=10000,
# ):

#     xopt = optim.minimize(
#         fun=T_c_cost,
#         x0=x0,
#         bounds=bounds,
#         args=(rho_ls, rho_gs, temps),
#         options=dict(maxiter=maxiter),
#         method="Nelder-Mead",
#     )
#     A, Tc_fit = xopt.x

#     return A, Tc_fit


# def rho_c(A, T):

#     B = A[0]
#     rho_c_fit = A[1]
#     Tc = A[2]

#     return rho_c_fit + B * (Tc - T)


# def rho_c_cost(A, rho_l, rho_g, T):
#     err = np.linalg.norm(((rho_l + rho_g) / 2) - rho_c(A, T)) ** 2
#     return err


# def fit_rho_c(
#     rho_ls,
#     rho_gs,
#     temps,
#     bounds=Bounds(lb=[0, 200, 300], ub=[200, 500, 500]),
#     maxiter=10000,
#     x0=[1, 300, 400],
# ):

#     xopt = optim.minimize(
#         fun=rho_c_cost,
#         x0=x0,
#         bounds=bounds,
#         args=(rho_ls, rho_gs, temps),
#         options=dict(maxiter=maxiter),
#         method="Nelder-Mead",
#     )

#     B, rho_c_fit, Tc = xopt.x

#     return B, rho_c_fit, Tc


# def Tc_rhoc_cost(fit_params, rho_ls, rho_gs, temps):

#     A = fit_params[:2]  # Tc -> A, Tc
#     B = fit_params[2:]  # rhoc -> B, rhoc
#     Tc = A[1]
#     B = np.concatenate((B, [Tc]))
#     err_Tc = T_c_cost(A, rho_ls, rho_gs, temps)
#     err_rhoc = rho_c_cost(B, rho_ls, rho_gs, temps)

#     return err_Tc + err_rhoc


# def fit_Tc_rhoc(
#     rho_ls,
#     rho_gs,
#     temps,
#     Tc_bounds=[[0, 300], [200, 500]],
#     rhoc_bounds=[[0, 100], [200, 600]],
#     A0=100,
#     Tc0=400,
#     B0=1,
#     rhoc0=300,
#     maxiter=10000,
# ):

#     lb = []
#     lb.extend(Tc_bounds[0])
#     lb.extend(rhoc_bounds[0])
#     ub = []
#     ub.extend(Tc_bounds[1])
#     ub.extend(rhoc_bounds[1])
#     bounds = Bounds(lb=lb, ub=ub)

#     A0, Tc0 = fit_T_c(
#         rho_ls,
#         rho_gs,
#         temps,
#         bounds=Bounds(lb=Tc_bounds[0], ub=Tc_bounds[1]),
#         x0=[A0, Tc0],
#         maxiter=maxiter,
#     )

#     rhoc_bounds[0].append(Tc_bounds[0][-1])  # Add Tc bounds
#     rhoc_bounds[1].append(Tc_bounds[1][-1])

#     B0, rhoc0, Tc0 = fit_rho_c(
#         rho_ls,
#         rho_gs,
#         temps,
#         bounds=Bounds(lb=rhoc_bounds[0], ub=rhoc_bounds[1]),
#         x0=[B0, rhoc0, Tc0],
#         maxiter=maxiter,
#     )

#     x0 = np.array([A0, Tc0, B0, rhoc0])

#     xopt = optim.minimize(
#         fun=Tc_rhoc_cost,
#         x0=x0,
#         bounds=bounds,
#         args=(rho_ls, rho_gs, temps),
#         method="Nelder-Mead",
#         options=dict(maxiter=maxiter),
#     )

#     A, Tc, B, rhoc = xopt.x
#     return Tc, rhoc
