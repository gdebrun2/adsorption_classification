import numpy as np
import numba as nb
import gc
import utils
import correlation


@nb.njit
def get_upper_tri_index(n: int) -> np.ndarray:
    indices = np.zeros((n * (n - 1) // 2, 2), dtype=np.int32)
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            indices[index] = i, j
            index += 1
    return indices


def generate_coord_num(
    radii: list,
    df: dict,
    start: int = 0,
    end: int = -1,
    method: str = "sparse",
    to_print: bool = False,
) -> None:
    if end == -1:
        end = df["nt"]

    nrad = len(radii)
    radii = np.array(radii).flatten()
    nmol = df["nmolecule"]

    if to_print:
        print("Generating Coordination number...", end="")

    if method == "full":
        if "distance" not in list(df["molecule"].keys()):
            distances = distance(df, method="full")
        else:
            distances = df["molecule"]["distance"][start:end]

        coord = np.zeros((nrad, df["nt"], df["nmolecule"]), dtype=np.int16)
        for idx, radius in enumerate(radii):
            coord[idx][start:end] = np.sum(distances < radius, axis=1) - 1
        del distances

    elif method == "sparse":

        distances = distance(df, method="sparse")
        coord = generate_coord_num_sparse(
            nmol, np.array(radii, dtype=float), distances
        ).astype(np.int16)

    dr = 2
    correlation.pair_correlation(df, dr, distances=distances)
    del distances
    gc.collect()
    for i, radius in enumerate(radii):

        if start != 0 or end != df["nt"]:
            df["molecule"][f"coordination_{radius}"] = np.zeros(
                (df["nt"], df["nmolecule"]), dtype=np.int16
            )
            df["molecule"][f"coordination_{radius}"][start:end] = coord[i]

        else:
            df["molecule"][f"coordination_{radius}"] = coord[i]

    del coord
    gc.collect()

    if to_print:
        print("Done")

    return None


@nb.njit
def distance_mol(dist: np.ndarray, mol_idx: int, ind: np.ndarray) -> np.ndarray:
    i = ind[:, 0]
    j = ind[:, 1]
    dist_mol_idx = np.where((i == mol_idx) | (j == mol_idx))[0]
    dist_mol = dist[:, dist_mol_idx]

    return dist_mol


@nb.njit(parallel=True)
def generate_coord_num_sparse(
    nmol: int, radii: list, distances: np.ndarray
) -> np.ndarray:

    nt = distances.shape[0]
    nrad = radii.shape[0]
    coord = np.zeros((nrad, nt, nmol), dtype=np.int32)
    ind = get_upper_tri_index(nmol)

    for rad_i in nb.prange(nrad):
        radius = radii[rad_i]
        for mol_idx in range(nmol):
            dist_mol = distance_mol(distances, mol_idx, ind)
            coord[rad_i, :, mol_idx] = np.sum(dist_mol <= radius, axis=1)

    return coord


@nb.njit(parallel=True)
def distance_sparse(R: np.ndarray, L: np.ndarray) -> np.ndarray:
    nt, n, _ = R.shape
    ndistances = n * (n - 1) // 2  # number of unique distances
    D = np.zeros((nt, ndistances), dtype=np.float64)
    for t in nb.prange(nt):
        index = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_squared = np.sum((utils.pbc(R[t, i] - R[t, j], L)) ** 2)
                D[t, index] = d_squared
                index += 1

    D = np.sqrt(D)
    return D


def distance_full(R: np.ndarray, L: np.ndarray) -> np.ndarray:
    ndim = R.ndim
    # One timestep
    if ndim == 2:
        dist = R[:, np.newaxis, :] - R
        dist = utils.pbc(dist, L)
        dist = np.linalg.norm(dist, axis=2)

    # All timesteps
    elif ndim == 3:
        dist = R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]
        dist = utils.pbc(dist, L)
        dist = np.linalg.norm(dist, axis=3)

    return dist


def distance(df: dict, method: str = "sparse") -> np.ndarray:
    """
    Compute distance table

    Args:
        R (np.array) : particle positions, shape (n, 3) or (nt, n, 3)
        L (np.array): side lengths of simulation box (3, )
        method: full distance table or flat distance arrays
    Returns:
        distance_table (np.array): distance table, shape (n, n) or (nt, n, n)
        or
        distance (np.array): shape (nt, n * (n -1) //2)

    """
    R = utils.get_pos(df)
    L = utils.get_L(df)
    ndim = R.ndim

    if method == "sparse" and ndim == 3:
        dist = distance_sparse(R, L)

    elif method == "full":
        dist = distance_full(R, L)

    return dist
