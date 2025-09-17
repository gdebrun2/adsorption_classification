import numba as nb
import numpy as np
import utils
import distance
from numpy.linalg import norm
import gc


def get_kvecs(n, L, step=1):
    """
    Calculate k vectors commensurate with a rectangular box.

    Consider only k vectors in the all-positive octant of reciprocal space.
        Plane wave energy eigenvalues are given by E = hbar^2 k^2 / 2m
        where k = |k| = 2pin/L is the magnitude of the k vector
        and n = (nx, ny, nz) is the set of integers that define the k vector
        and L are the side lengths of the simulation cell
        Energy values are negatively degenerate, so only consider the positive octant

    Args:
        n : maximum value for nx, ny, nz; n+1 is number of k-points along each axis
            or array of n values
        L : side lengths of cell

    Returns:
        array of shape (nk, ndim): k vectors
    """
    if isinstance(n, int):
        n = np.arange(1, n + 1, step)

    n = np.vstack(np.meshgrid(n, n, n)).reshape(3, -1).T
    kvecs = 2 * np.pi / L * n
    return kvecs.astype(np.float32)


def get_kmags(kvecs):
    kmags = norm(kvecs, axis=1)
    return kmags


def kmags_to_wavelength(unique_kmags):
    unique_wavelength = 2 * np.pi / unique_kmags

    return unique_wavelength


def get_kmask(kmags):
    """
    Get unique k-vector magnitudes and store the indices
    of the kvecs with the same magnitude

    Args:
        kmags (np.array): array of k-vector magnitudes
    Returns:
        unique_kmags (np.array): array of unique k-vector magnitudes
        kmask (list): list of masks for each unique k-vector magnitude
    """
    unique_kmags = np.unique(kmags)
    kmask = []
    for kmag in unique_kmags:
        mask = np.where(kmags == kmag)
        kmask.append(mask)

    return unique_kmags, kmask


def sk_init(df, n):
    pe_mol = np.sum(df["molecule"]["pe"], axis=1)
    kappa, start, ac = actime(pe_mol)
    bounds = df["bounds"][:, 1]
    kvecs = get_kvecs(n, bounds)
    kmags = get_kmags(kvecs)
    unique_kmags, kmask = get_kmask(kmags)

    return kvecs, unique_kmags, kmask, start


@nb.njit
def calc_rhok(kvecs, pos):
    """
    Calculate the fourier transform of particle density distribution.

    Args:
        kvecs (np.array): array of k-vectors, shape (nk, ndim)
        pos (np.array): particle positions, shape (ndim, Natom)
    Returns:
        array of shape (nk,): fourier transformed density rho_k
    """
    return np.sum(np.exp(-1j * (kvecs @ pos)), axis=-1)


@nb.njit
def calc_sk(kvecs, pos):
    """
    Calculate the structure factor S(k).

    Args:
        kvecs (np.array): array of k-vectors, shape (nk, ndim)
        pos (np.array): particle positions, shape (ndim, natom)
    Returns:
        array of shape (nk,): structure factor s(k)
    """
    N = pos.shape[1]
    I = calc_rhok(kvecs, pos) * calc_rhok(-kvecs, pos)
    sk = I / N
    return sk.real.astype(np.float32)


@nb.njit(parallel=True)
def sk_time_average(r, kvecs, kmask, start, nt=-1):
    nmags = len(kmask)
    if nt == -1:
        nt = r.shape[2]
    unique_sks = np.zeros((nt, nmags))
    for t in nb.prange(start, nt):
        sk = calc_sk(kvecs, r[:, :, t])
        unique_sk = np.zeros(nmags)
        for i in range(nmags):
            mask = kmask[i]
            unique_sk[i] = np.mean(sk[mask].real)

        unique_sks[t] = unique_sk

    # sk_avg = np.mean(unique_sks, axis=0) # numba doesnt support axis argument
    sk_avg = np.sum(unique_sks, axis=0) / unique_sks.shape[0]
    return sk_avg


def ssf_binned(df, n, sk, dw=None, dk=None):
    if dw and dk or (not dw and not dk):
        raise ValueError("Either dw or dk must be specified")

    kvecs, kmags, kmask, start = sk_init(df, n)

    if dw:
        wavelengths = kmags_to_wavelength(kmags)
        bins = np.arange(wavelengths.min(), wavelengths.max() + dw, dw)
        x = wavelengths

    elif dk:
        bins = np.arange(kmags.min(), kmags.max() + dk, dk)
        x = kmags

    bin_centers = (bins[1:] + bins[:-1]) / 2
    sk_bin = np.zeros(bins.shape[0] - 1)
    for i in range(1, bins.shape[0]):
        mask = np.where((x >= bins[i - 1]) & (x < bins[i]))
        if sk[mask].shape[0] == 0:  # if no values in bin
            sk_bin[i - 1] = np.nan
        else:  # otherwise, take mean of all values in bin
            sk_bin[i - 1] = np.mean(sk[mask])
    nan_mask = np.isnan(sk_bin)
    bin_centers = bin_centers[~nan_mask]
    sk_bin = sk_bin[~nan_mask]
    return bin_centers, sk_bin


def actime(E):
    """
    Calculate the autocorrelation time of a time series E.
    At a given time lag k, the autocorrelation is given by
    ac(k) = 1 / (nt - k) * sum_{i=0}^{nt-k} (E_i - mu) * (E_{i+k} - mu)

    In words, this represents the correlation between the first
    nt - k elements of the time series and the next nt - k elements.
    At k = 0, the autocorrelation is exactly 1

    The sample mean and variance are used instead of the mean
    and variance of the two periods due to the assumption of
    ergodicity and thus stationarity.

    Args:
        E : 1d array of time series data
    Returns:
        tau : autocorrelation time
        tcutoff : index at which the autocorrelation becomes negative
        ac : array of autocorrelation values

    Note:
        Can also be calculated using an FFT
    """
    mu = np.mean(E)
    std = np.std(E, ddof=1)
    nt = E.shape[0]
    lags = np.arange(0, nt, 1)
    ac = np.zeros(nt)

    for lag in lags:
        ac[lag] = np.mean((E[: nt - lag] - mu) * (E[lag:] - mu)) / std**2

    tcutoff = np.where(ac <= 0)[0][0] if (ac <= 0).any() else 0
    tau = 1 + 2 * np.sum(ac[1:tcutoff])
    return tau, tcutoff, ac


@nb.njit
def _pair_correlation(distances, nmol, dr, L):
    """
    Calculate the pair correlation function g(r) over all t

    Args:
        distances (np.array): 2d array of unique pair distances
        dr (float): size of bins
        L (np.array): box dimensions
    Returns:
        g: pair correlation function
        r: bin centers
    """

    nt = distances.shape[0]
    distances = distances.flatten()
    bins = np.arange(distances.min(), distances.max() + dr, dr)
    r = (bins[:-1] + bins[1:]) / 2
    counts = np.histogram(distances, bins=bins)[0] / nt
    V = np.prod(L)
    rho = nmol / V
    shell_volume = 4.0 * np.pi / 3.0 * ((r + dr / 2) ** 3 - (r - dr / 2) ** 3)
    Nideal = rho * shell_volume
    # Nideal_pairs = Nideal * (nmol - 1) / 2
    Nideal_pairs = Nideal * nmol
    g = counts / Nideal_pairs
    return g, r


def pair_correlation(df, dr, distances=None):

    if distances is None:
        distances = distance.distance(df)
    nmol = df["nmolecule"]
    L = utils.get_L(df)
    g, r = _pair_correlation(distances, nmol, dr, L)
    df["molecule"]["g"] = g
    df["molecule"]["r"] = r
    del distances
    gc.collect()

    return g, r


@nb.njit
def _calc_vaf0(v, t):
    # vacf = np.sum(np.einsum("ij,ij->i", v[0], v[t])) / v.shape[1]
    vacf = np.sum(np.sum(v[0] * v[t], axis=1)) / v.shape[1]
    return vacf


@nb.njit
def calc_vacf0(v):
    nt = v.shape[0]
    vacf = np.zeros(nt)
    for t in range(nt):
        vacf[t] = _calc_vaf0(v, t)
    return vacf


def diffusion_constant(vacf, dt):
    """
    Calculate the diffusion constant from the
    velocity-velocity auto-correlation function (vacf).

    Args:
        vacf (np.array): shape (nt,), vacf sampled at
        nt time steps from t=0 to nt*dt
        dt (float): time step in seconds
    Returns:
        float: the diffusion constant calculated from the vacf.
    """
    return np.trapz(vacf, dx=dt) / 3


def distance(x, y):
    """
    Distance correlation looks for correlation in the
    pairwise distances of two vectors (e.g. compares the
    distribution of (X_i - X_j) to (Y_i - Y_j)

    0 -> No correlation
    1 -> Perfect correlation

    Measures linear and nonlinear correlation.
    """

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    numerator = (x - mu_x).T @ (y - mu_y)
    denominator = norm(x - mu_x) * norm(y - mu_y)

    return numerator / denominator


def cosine(x, y):

    return np.dot(x, y) / (norm(x) * norm(y))


def pearson(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.corrcoef(x, y)[1, 0]


def corr_matrix(df, features, method="pearson", N=10000):
    nfeats = len(features)
    mat = np.zeros((nfeats, nfeats))
    func_map = {
        "pearson": pearson,
        "distance": distance,
        "mic": mic,
        "cosine": cosine,
        "hhg": hhg,
        "rdc": rdc,
    }
    lag = utils.get_lag(df, features=features)
    values = [utils.parse(df, feat)[lag:].flatten() for feat in features]
    X = np.array(values).T
    for i in range(nfeats):
        x = X[:, i]
        for j in range(i, nfeats):
            y = X[:, j]
            if method == "hhg" or method == "mic":
                corr = func_map[method](x, y, N)
            else:
                corr = func_map[method](x, y)

            mat[i, j] = corr
            mat[j, i] = corr

    return mat


######################################### Archive ##########################################

# from minepy import MINE
# from rdc import rdc as RDC
# from hyppo.independence import HHG

# def get_kvecs(
#     lbox,
#     ntheta=10,
#     Nphi=10,
#     mag_min=0,
#     mag_max=5,
#     mag_step=1,
# ):
#     """
#     Calculate k vectors commensurate with a simulation box.

#     Args:
#         ntheta: number of angles in theta direction (0 to pi)
#         Nphi: number of angles in phi direction (0 to 2pi)
#         mag_min: minimum magnitude of k vector
#         mag_max: maximum magnitude of k vector
#         lbox : side lengths of box

#     Returns:
#         array of shape (nk, ndim): collection of k vectors
#     """

#     theta = np.linspace(0, np.pi, ntheta, endpoint=False)
#     phi = np.linspace(0, 2 * np.pi, Nphi + 1, endpoint=False)[1:]
#     kmags = np.arange(mag_min, mag_max + mag_step, mag_step)
#     theta_grid, phi_grid, mag_grid = np.meshgrid(theta, phi, kmags, indexing="ij")
#     kx = mag_grid * np.sin(theta_grid) * np.cos(phi_grid)
#     ky = mag_grid * np.sin(theta_grid) * np.sin(phi_grid)
#     kz = mag_grid * np.cos(theta_grid)

#     kvecs = np.stack((kx, ky, kz), axis=-1)

#     kvecs = kvecs.reshape(-1, 3)
#     kvecs = kvecs[ntheta - 1 :]

#     return kvecs


# def mic(x, y, N=10000):
#     """nonlinear correlation coefficient with a lot of criticism"""
#     idx = np.random.choice(N, size=N)
#     mine = MINE()
#     x = x.flatten()[idx]
#     y = y.flatten()[idx]
#     mine.compute_score(x, y)

#     return mine.mic()


# def rdc(x, y):
#     """nonlinear correlation coefficient"""
#     x = x.flatten()
#     y = y.flatten()
#     return RDC(x, y)


# def hhg(x, y, N=10000):
#     """nonlinear correlation test"""
#     idx = np.random.choice(N, size=N)
#     x = x.flatten()[idx]
#     y = y.flatten()[idx]
#     stat, pvalue = HHG().test(x, y, workers=-1, auto=True)

#     return pvalue
