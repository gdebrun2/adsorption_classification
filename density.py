import numpy as np
import numba as nb
import utils
import gc
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import time


@nb.njit(parallel=True)
def get_min_max(z):
    nt = z.shape[0]
    max_neg = np.zeros(nt)
    max_pos = np.zeros(nt)
    for t in nb.prange(nt):
        zt = z[t]
        z_pos = zt[zt >= 0]
        z_neg = zt[zt < 0]
        max_pos[t] = np.max(z_pos)
        max_neg[t] = np.max(np.abs(z_neg))
    return np.min(max_pos), np.min(max_neg)


def density(
    df,
    bin_width=2,  # Angstroms
    z=None,
    mode="atom",
    hist_range=None,
    time_avg=True,
    phase_mask=None,
    absval=True,
    center=True,
    offset=True,
    actime=False,
    auto_range=True,
    std=False,
    norm="mass",
    t=None,
    start=0,
):

    if z is None:
        z = utils.parse(df, "z", mode)
    if offset:
        lower_mask = df[mode]["lower_mask"]
        z[lower_mask] -= df["offset"]

    if "cluster_vars" in list(df.keys()):
        start = utils.get_lag(df, start, actime)

    if center:
        com = np.mean(df["atom"]["z"], axis=1)
        z -= com[:, np.newaxis]
        L = utils.get_L(df)
        z = utils.pbc(z, L[-1])
    if t is None:
        z = z[start:]
        nt = z.shape[0]
    else:
        z = z[t]
        nt = 1
    if auto_range:
        if t is None:
            if not absval:
                zmin = np.max(np.min(z, axis=1))  # lowest z bin present for all t
                zmax = np.min(np.max(z, axis=1))  # highest z bin present for all t
            else:
                zmax_pos, zmax_neg = get_min_max(z)
                zmin = np.max(np.min(np.abs(z), axis=1))
                zmax = zmax_pos if zmax_pos <= zmax_neg else zmax_neg

        else:
            if not absval:
                zmin = z.min()
                zmax = z.max()
            else:
                zmin = np.min(np.abs(z))
                zmax1 = np.max(np.abs(z[z < 0]))
                zmax2 = np.max(z[z > 0])
                zmax = zmax1 if zmax1 < zmax2 else zmax2
    if absval:
        z = np.abs(z)
    if hist_range is None and not auto_range:
        if not absval:
            zmax = df["bounds"][-1, -1]
            zmin = df["bounds"][-1, 0]
        else:
            zmax = df["bounds"][-1, -1]
            zmin = z.min()

    if hist_range is None:
        hist_range = (zmin, zmax)
    else:
        zmin, zmax = hist_range

    nbins = int(np.round((zmax - zmin) / bin_width, 0))

    if nt == 1:
        time_avg = False

    phase = None
    if (time_avg or nt == 1) and not std and phase_mask is None and norm != "mass":

        counts, bins = np.histogram(z, bins=nbins, range=hist_range)
        zbin = (bins[:-1] + bins[1:]) / 2
        density = counts / nt

        density = normalize_density(
            df,
            density,
            z,
            norm,
            phase_mask,
            mode,
            zbin,
            start,
            bins,
            nt,
            phase,
        )
        if absval:
            density /= 2
        return density, zbin

    else:
        bins = np.linspace(hist_range[0], hist_range[1], nbins + 1, endpoint=True)
        zbin = (bins[1:] + bins[:-1]) / 2

        if phase_mask is not None:
            phase = utils.parse(df, "phase", mode=mode)
            if t is None:
                phase = phase[start:]
            else:
                phase = phase[t]
            mask = phase == phase_mask
            if nt == 1:
                density = density_nt_phase(
                    z.reshape(1, -1), nt, nbins, hist_range, mask.reshape(1, -1)
                )
            else:
                density = density_nt_phase(z, nt, nbins, hist_range, mask)
        else:
            density = density_nt(z, nt, nbins, hist_range)
        density = normalize_density(
            df,
            density,
            z,
            norm,
            phase_mask,
            mode,
            zbin,
            start,
            bins,
            nt,
            phase,
        )
        if absval:
            density /= 2
        if nt == 1:
            density = density.flatten()
        err = None
        if std:
            err = np.std(density, axis=0)
        if time_avg:
            density = np.mean(density, axis=0)
        if auto_range:
            mask = density > 0
            density = density[mask]
            zbin = zbin[mask]
            if std:
                err = err[mask]

        return density, zbin, err


def normalize_density(
    df,
    density,
    z,
    norm,
    phase_mask,
    mode,
    zbin,
    start,
    bins,
    nt,
    phase,
):

    nt = density.shape[0]
    nbins = bins.shape[0] - 1
    if norm == "count":
        return density
    if norm == "prob" or norm == "percent":
        if mode == "atom":
            n = df["natom"] - df["nmetal"]
        else:
            n = df["nmolecule"]

        density /= n

    elif norm == "mass":

        L = utils.get_L(df)
        dz = np.abs(zbin[1] - zbin[0])
        slice_volume = dz * L[0] * L[1] * 1e-10**3
        if mode == "atom":

            atom_masses = df["atom"]["mass"][df["non_metal"]] * df["AMU_TO_KG"]
            bin_indices = np.digitize(z, bins) - 1  # (nt, natom)

            if phase_mask is not None:
                phase_filter = phase == phase_mask
            else:
                phase_filter = np.ones_like(z, dtype=bool)

            if nt == 1:
                bin_mass = atomic_bin_mass(
                    nbins,
                    nt,
                    bin_indices.reshape(1, -1),
                    atom_masses,
                    phase_filter.reshape(1, -1),
                )
            else:
                bin_mass = atomic_bin_mass(
                    nbins, nt, bin_indices, atom_masses, phase_filter
                )

            density = bin_mass / slice_volume

        else:
            mol_weight = df["molecule"]["mass"] * df["AMU_TO_KG"]
            density = density * mol_weight / slice_volume

    return density


@nb.njit(parallel=True)
def atomic_bin_mass(nbins, nt, bin_indices, atom_masses, phase_mask):
    bin_mass = np.zeros((nt, nbins))
    for i in nb.prange(nbins):
        bin_filter = bin_indices == i  # every atom that belongs to bin i (nt, natom)
        for t in nb.prange(nt):
            bin_i_t = bin_filter[t]
            bin_mass[t][i] = np.sum(
                atom_masses[phase_mask[t] & bin_i_t]
            )  # map atoms to masses and sum
    return bin_mass


@nb.njit(parallel=True)
def density_nt_phase(z, nt, nbins, hist_range, mask):
    density = np.zeros((nt, nbins))
    for t in nb.prange(nt):

        z_t = z[t][mask[t]]
        counts, _ = np.histogram(z_t, bins=nbins, range=hist_range)
        density[t] = counts

    return density


@nb.njit(parallel=True)
def density_nt(z, nt, nbins, hist_range):
    density = np.zeros((nt, nbins))
    for t in nb.prange(nt):

        counts, _ = np.histogram(z[t], bins=nbins, range=hist_range)
        density[t] = counts

    return density


# ----------------------------------------------------#


def density_one_step(
    t,
    pos_mol,
    pos_atom,
    atom_masses,
    molecule_ids,
    L,
    radii,
    workers_tree=-1,
    n_jobs_radii=1,
):
    """
    Return densities for ONE time step (shape (1, nmol, nrad)).
    Uses a single sparse_distance_matrix up to r_max, then filters per radius.
    """
    nmol = pos_mol.shape[1]
    atoms = np.mod(pos_atom[t], L)
    centres = np.mod(pos_mol[t], L)
    nrad = radii.shape[0]
    V = (4.0 / 3.0) * np.pi * radii**3

    # Build both trees once
    tree_atoms = cKDTree(atoms, boxsize=L)
    tree_centres = cKDTree(centres, boxsize=L)

    # One pass up to the largest radius
    r_max = float(np.max(radii))
    # COOrdinate sparse format gives us row/col/data arrays directly
    sdm = tree_centres.sparse_distance_matrix(
        tree_atoms, max_distance=r_max, output_type="coo_matrix"
    )
    rows = sdm.row.astype(np.int32)  # centre (molecule) indices
    cols = sdm.col.astype(np.int32)  # atom indices
    dists = sdm.data.astype(np.float64)

    # Zero out contributions from same-molecule atoms
    same = molecule_ids[cols] == rows
    weights_all = np.where(same, 0.0, atom_masses[cols])

    def dens_for_radius(ri):
        r = radii[ri]
        mask = dists <= r
        if not np.any(mask):
            return np.zeros(nmol, dtype=np.float64)
        mass_sum = np.bincount(rows[mask], weights=weights_all[mask], minlength=nmol)
        return mass_sum / V[ri]

    if n_jobs_radii == 1:
        dens_list = [dens_for_radius(ri) for ri in range(nrad)]
    else:
        dens_list = Parallel(n_jobs=n_jobs_radii, prefer="threads", batch_size="auto")(
            delayed(dens_for_radius)(ri) for ri in range(nrad)
        )

    densities = np.stack(dens_list, axis=-1)[None, :, :]  # (1, nmol, nrad)
    return densities


def _atomic_density(
    pos_mol,
    pos_atom,
    atom_masses,
    molecule_ids,
    L,
    radii,
    n_jobs=-1,
    batch_size="auto",
    workers=-1,
    prefer="processes",
    n_jobs_radii=1,
):
    nt = pos_mol.shape[0]

    def step(t):
        return density_one_step(
            t,
            pos_mol,
            pos_atom,
            atom_masses,
            molecule_ids,
            L,
            radii,
            workers_tree=workers,
            n_jobs_radii=n_jobs_radii,
        )

    dens = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer=prefer)(
        delayed(step)(t) for t in range(nt)
    )
    ret = np.vstack(dens) / (1e-10**3)
    return ret


def atomic_density(
    df,
    radii,
    nt=None,
    n_jobs=32,
    batch_size="auto",
    to_print=False,
    workers=-1,
    prefer="processes",
    n_jobs_radii=1,
):
    radii = np.asarray(radii)
    if to_print:
        s = time.time()
        print("\nCalculating atomic density...", end="")

    if nt is None:
        nt = df["molecule"]["x"].shape[0] + 1
    pos_atom = utils.get_pos(df, mode="atom")[:nt]
    pos_mol = utils.get_pos(df)[:nt]
    molecule_ids = df["atom"]["molecule_id"]
    amu = df["AMU_TO_KG"]
    atom_masses = df["atom"]["mass"] * amu
    L = utils.get_L(df)

    dens = _atomic_density(
        pos_mol,
        pos_atom,
        atom_masses,
        molecule_ids,
        L,
        radii,
        n_jobs=n_jobs,
        batch_size=batch_size,
        workers=workers,
        prefer=prefer,
        n_jobs_radii=n_jobs_radii,
    )

    for ri in range(dens.shape[-1]):
        df["molecule"][f"density_{radii[ri]}"] = dens[:, :, ri] / 1000

    del pos_atom, pos_mol
    gc.collect()

    if to_print:
        e = time.time()
        dt = e - s
        print(f" Done in {dt:.2f}s")

    return None



######################################### Archive ##########################################


# def density_one_step(
#     t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radii, workers
# ):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radii**3
#     # --- wrap coords ---
#     atoms = np.mod(pos_atom[t], L)
#     centres = np.mod(pos_mol[t], L)

#     tree = cKDTree(atoms, boxsize=L)
#     densities = []
#     for ri, radius in enumerate(radii):

#         lists = tree.query_ball_point(
#             centres,
#             r=radius,
#             workers=workers,
#         )

#         lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#         owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#         neigh = np.concatenate(lists)

#         same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#         weights = np.where(same, 0.0, atom_masses[neigh])
#         mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#         dens = mass_sum / V[ri]
#         densities.append(dens)

#     return np.array(densities).T


# def _atomic_density(
#     pos_mol,
#     pos_atom,
#     atom_masses,
#     molecule_ids,
#     L,
#     radii,
#     n_jobs=-1,
#     batch_size="auto",
#     workers=-1,
#     prefer="processes",
# ):
#     nt = pos_mol.shape[0]

#     def step(t):
#         return density_one_step(
#             t,
#             pos_mol,
#             pos_atom,
#             atom_masses,
#             molecule_ids,
#             L,
#             radii,
#             workers,
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer=prefer)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)


# def atomic_density(
#     df,
#     radii,
#     nt=None,
#     n_jobs=32,
#     batch_size="auto",
#     to_print=False,
#     workers=-1,
#     prefer="processes",
# ):
#     radii = np.asarray(radii)
#     if to_print:
#         s = time.time()
#         print("\nCalculating atomic density...", end="")

#     if nt is None:
#         nt = df["molecule"]["x"].shape[0] + 1
#     pos_atom = utils.get_pos(df, mode="atom")[:nt]
#     pos_mol = utils.get_pos(df)[:nt]
#     molecule_ids = df["atom"]["molecule_id"]
#     amu = df["AMU_TO_KG"]
#     atom_masses = df["atom"]["mass"] * amu
#     L = utils.get_L(df)

#     dens = _atomic_density(
#         pos_mol,
#         pos_atom,
#         atom_masses,
#         molecule_ids,
#         L,
#         radii,
#         n_jobs=n_jobs,
#         batch_size=batch_size,
#         workers=workers,
#         prefer=prefer,
#     )

#     for ri in range(dens.shape[-1]):
#         df["molecule"][f"density_{radii[ri]}"] = dens[:, :, ri] / 1000

#     del pos_atom, pos_mol
#     gc.collect()

#     if to_print:
#         e = time.time()
#         dt = e - s
#         print(f" Done in {dt:.2f}s")

#     return None


# ---- your fast single-step kernel -----------------------------
# def density_one_step(
#     t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, workers
# ):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     atoms = np.mod(pos_atom[t], L)
#     centres = np.mod(pos_mol[t], L)

#     tree = cKDTree(atoms, boxsize=L)
#     lists = tree.query_ball_point(
#         centres,
#         r=radius,
#         workers=workers,
#     )

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     dens = mass_sum / V

#     return dens


# def _atomic_density(
#     pos_mol,
#     pos_atom,
#     atom_masses,
#     molecule_ids,
#     L,
#     radius,
#     n_jobs=-1,
#     batch_size=100,
#     workers=-1,
#     prefer="processes",
# ):
#     nt = pos_mol.shape[0]

#     def step(t):
#         return density_one_step(
#             t,
#             pos_mol,
#             pos_atom,
#             atom_masses,
#             molecule_ids,
#             L,
#             radius,
#             workers,
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer=prefer)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)


# def atomic_density(
#     df,
#     radii,
#     nt=None,
#     n_jobs=32,
#     batch_size=100,
#     to_print=False,
#     workers=-1,
#     prefer="processes",
# ):

#     if to_print:
#         s = time.time()
#         print("\nCalculating atomic density...", end="")

#     if nt is None:
#         nt = df["molecule"]["x"].shape[0] + 1
#     pos_atom = utils.get_pos(df, mode="atom")[:nt]
#     pos_mol = utils.get_pos(df)[:nt]
#     molecule_ids = df["atom"]["molecule_id"]
#     amu = df["AMU_TO_KG"]
#     atom_masses = df["atom"]["mass"] * amu
#     L = utils.get_L(df)

#     for radius in radii:

#         dens = _atomic_density(
#             pos_mol,
#             pos_atom,
#             atom_masses,
#             molecule_ids,
#             L,
#             radius,
#             n_jobs=n_jobs,
#             batch_size=batch_size,
#             workers=workers,
#             prefer=prefer,
#         )
#         df["molecule"][f"density_{radius}"] = dens / 1000

#     del pos_atom, pos_mol
#     gc.collect()

#     if to_print:
#         e = time.time()
#         dt = e - s
#         print(f" Done in {dt:.2f}s")

#     return None


# def atomic_density(df, radii):

#     pos_atom = utils.get_pos(df, mode="atom")
#     pos_mol = utils.get_pos(df)
#     mol_ids = df["atom"]["molecule_id"]
#     natom_per_molecule = df["natom_per_molecule"]
#     atom_mass = df["atom"]["mass"]
#     amu = df["AMU_TO_KG"]
#     dens = _atomic_density(pos_atom, pos_mol, mol_ids, radii, atom_mass, amu)
#     del pos_atom, pos_mol
#     gc.collect()

#     for i, d in enumerate(dens):

#         df["molecule"][f"atom_density_{radii[i]}"] = d

#     return None


# def atomic_density(df, radii, nt):

#     pos_atom = utils.get_pos(df, mode="atom")[:nt]
#     pos_mol = utils.get_pos(df)[:nt]
#     molecule_ids = df["atom"]["molecule_id"]
#     amu = df["AMU_TO_KG"]
#     atom_masses = df["atom"]["mass"] * amu

#     radii = np.array(radii, dtype=int)
#     L = utils.get_L(df)

#     dens = _atomic_density(pos_mol, pos_atom, atom_masses, molecule_ids, L, radii)
#     del pos_atom, pos_mol, atom_masses
#     gc.collect()

#     for i, d in enumerate(dens):

#         df["molecule"][f"density_{radii[i]}"] = d

#     return None


# def _wrap_into_box(x, L):
#     """Return positions mapped into the range [0, L) for each axis."""
#     return np.mod(x, L)  # works for negative values too


# def _atomic_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in range(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = cKDTree(atoms_in_box, boxsize=L)
#         nbrs = tree.query_ball_point(mols_in_box, r=radius, workers=-1)

#         # --- 3. convert neighbour lists to densities ------------------------
#         for m_id, idx in enumerate(nbrs):
#             if idx:  # non-empty
#                 foreign = molecule_ids[idx] != m_id  # mask out own atoms
#                 mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#                 dens[t, m_id] = mass_sum / V
#             else:
#                 dens[t, m_id] = 0
#     del atoms_in_box, mols_in_box, tree, nbrs, foreign, mass_sum
#     gc.collect()
#     return dens


# def _atomic_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in range(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = cKDTree(atoms_in_box, boxsize=L)
#         nbrs = tree.query_ball_point(mols_in_box, r=radius, workers=-1)

#         dens[t] = neighbors_to_density(nbrs, molecule_ids, atom_masses, V)

#     del atoms_in_box, mols_in_box, tree, nbrs, foreign, mass_sum
#     gc.collect()
#     return dens


# @nb.njit(parallel=True)
# def neighbors_to_density(nbrs, molecule_ids, atom_masses, V):

#     dens = np.empty(nbrs.size, dtype=np.float64)

#     for m_id in nb.prange(nbrs.size):
#         idx = nbrs[m_id]
#         if idx.any():  # non-empty
#             foreign = molecule_ids[idx] != m_id  # mask out own atoms
#             mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#             dens[m_id] = mass_sum / V
#         else:
#             dens[m_id] = 0

#     return dens


# @nb.njit
# def _atomic_density(pos_atom, pos_mol, mol_ids, radii, atom_mass, amu):

#     nrad = len(radii)
#     nt, nmol, _ = pos_mol.shape[0]
#     # natom = pos_atom.shape[1]

#     dens = np.zeros((nrad, nt, nmol))

#     for t in range(nt):

#         for i in range(nmol):

#             curr_pos = pos_mol[t, i]  # (3,)
#             curr_mol = np.where(mol_ids == i)
#             curr_atom_pos = pos_atom[t, ~curr_mol]  # (natom-nmolecule_per_atom, 3)
#             curr_atom_mass = atom_mass[~curr_mol]
#             diff = curr_atom_pos - curr_pos
#             dist = np.linalg.norm(diff)

#             for j in range(nrad):
#                 rad = radii[j]
#                 V = 4 / 3 * np.pi * rad**3
#                 in_sphere = dist <= rad
#                 mass_in_sphere = curr_atom_mass[in_sphere] * amu
#                 d = mass_in_sphere / V

#                 dens[j, t, i] = d

# import numpy as np
# from numba_kdtree import KDTree
# import math


# @nb.njit
# def _wrap_into_box(x, L):
#     """Return positions mapped into the range [0, L) for each axis."""
#     return np.mod(x, L)  # works for negative values too


# @nb.njit(parallel=True)
# def local_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in nb.prange(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = KDTree(atoms_in_box)
#         nbrs = tree.query_radius_parallel(mols_in_box, r=radius)

#         # --- 3. convert neighbour lists to densities ------------------------
#         for m_id, idx in enumerate(nbrs):
#             if idx.size > 0:  # non-empty
#                 foreign = molecule_ids[idx] != m_id  # mask out own atoms
#                 mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#                 dens[t, m_id] = mass_sum / V
#             else:
#                 dens[t, m_id] = 0

#     return dens

# import utils
# import numpy as np
# import numba as nb


# def atomic_density(df, radii, nt):

#     pos_atom = utils.get_pos(df, mode="atom")[:nt]
#     pos_mol = utils.get_pos(df)[:nt]
#     mol_ids = df["atom"]["molecule_id"]
#     atom_mass = df["atom"]["mass"]
#     amu = df["AMU_TO_KG"]
#     radii = np.array(radii, dtype=int)
#     L = utils.get_L(df)

#     dens = _atomic_density(pos_atom, pos_mol, mol_ids, radii, atom_mass, amu, L)
#     del pos_atom, pos_mol
#     gc.collect()

#     for i, d in enumerate(dens):

#         df["molecule"][f"density_{radii[i]}"] = d

#     return None


# @nb.njit(parallel=True, fastmath=True)
# def _atomic_density(pos_atom, pos_mol, mol_ids, radii, atom_mass, amu, L):

#     nrad = radii.shape[0]
#     nt, nmol, _ = pos_mol.shape

#     dens = np.zeros((nrad, nt, nmol))

#     for t in nb.prange(nt):

#         for i in range(nmol):

#             curr_pos = pos_mol[t, i]  # current molecule COM (3,)
#             curr_mol = mol_ids == i
#             curr_atom_pos = pos_atom[
#                 t, ~curr_mol
#             ]  # Atoms not in mol (natom - natom_per_molecule, 3)
#             curr_atom_mass = atom_mass[
#                 ~curr_mol
#             ]  # Atomic masses (natom - natom_per_molecule,)
#             diff = curr_atom_pos - curr_pos

#             diff = utils.pbc(diff, L)

#             dist = np.sqrt(np.sum(diff**2, axis=-1))

#             for j in range(nrad):
#                 rad = radii[j]
#                 V = 4 / 3 * np.pi * (rad * 1e-10) ** 3
#                 in_sphere = dist <= rad
#                 mass_in_sphere = np.sum(curr_atom_mass[in_sphere] * amu)
#                 d = mass_in_sphere / V
#                 dens[j, t, i] = d

#     return dens


# @nb.njit(inline="always")
# def pbc_fast(R, L):

#     R[:, 0] -= L[0] * np.round(R[:, 0] / L[0])
#     R[:, 1] -= L[1] * np.round(R[:, 1] / L[1])
#     R[:, 2] -= L[2] * np.round(R[:, 2] / L[2])


# @nb.njit(parallel=True, fastmath=False, cache=False)
# def _atomic_density(pos_atom, pos_mol, mol_ids, radii, atom_mass, amu, L):

#     nt, natom, _ = pos_atom.shape
#     _, nmol, _ = pos_mol.shape
#     nrad = radii.size
#     rad2 = radii * radii
#     invV = 3.0 / (4.0 * np.pi * (radii * 1e-10) ** 3)
#     atom_m = atom_mass * amu

#     dens = np.empty((nrad, nt, nmol), dtype=np.float64)

#     for t in nb.prange(nt):
#         a_pos = pos_atom[t]  # (natom,3)
#         coms = pos_mol[t, :]  # (nmolecule, 3)

#         has_neighbors = 0
#         for mol_idx in range(nmol):

#             com = coms[mol_idx]  # (3,)

#             diff = a_pos - com
#             pbc_fast(diff, L)

#             dist2 = (diff * diff).sum(axis=1)  # (natom,)
#             not_mol = mol_ids != mol_idx  # (natom,) bool

#             for k in range(nrad):
#                 inside = (dist2 <= rad2[k]) & not_mol
#                 if rad2[k] == 15**2 and inside.any():
#                     has_neighbors += 1
#                 mass = atom_m[inside].sum()
#                 dens[k, t, mol_idx] = mass * invV[k]
#         print(t, has_neighbors)

#     return dens


# import numpy as np
# from scipy.spatial import cKDTree


# @nb.njit
# def _wrap_into_box(x, L):
#     """Return positions mapped into the range [0, L) for each axis."""
#     return np.mod(x, L)  # works for negative values too


# def local_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in range(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = cKDTree(atoms_in_box, boxsize=L)
#         nbrs = tree.query_ball_point(mols_in_box, r=radius, workers=-1)

#         # --- 3. convert neighbour lists to densities ------------------------
#         for m_id, idx in enumerate(nbrs):
#             if idx:  # non-empty
#                 foreign = molecule_ids[idx] != m_id  # mask out own atoms
#                 mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#                 dens[t, m_id] = mass_sum / V
#             else:
#                 dens[t, m_id] = 0

#     return dens

# import numpy as np
# from numba_kdtree import KDTree
# import math


# @nb.njit
# def _wrap_into_box(x, L):
#     """Return positions mapped into the range [0, L) for each axis."""
#     return np.mod(x, L)  # works for negative values too


# @nb.njit(parallel=True)
# def local_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in nb.prange(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = KDTree(atoms_in_box)
#         nbrs = tree.query_radius_parallel(mols_in_box, r=radius)

#         # --- 3. convert neighbour lists to densities ------------------------
#         for m_id, idx in enumerate(nbrs):
#             if idx.size > 0:  # non-empty
#                 foreign = molecule_ids[idx] != m_id  # mask out own atoms
#                 mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#                 dens[t, m_id] = mass_sum / V
#             else:
#                 dens[t, m_id] = 0

#     return dens


# from numba.typed import List


# def local_density(
#     pos_mol,  # (nt, nmol, 3)
#     pos_atom,  # (nt, natom, 3)
#     atom_masses,  # (natom,)
#     molecule_ids,  # (natom,)  -- 0,1,2,...
#     L,  # (3,)
#     radius,
# ):
#     """
#     Local atomic density dens[t, i] excluding atoms that belong to molecule i
#     using a periodic KD-tree
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * np.pi * (radius * 1e-10) ** 3
#     nrad = radius.size
#     dens = np.empty((nt, nmol), dtype=np.float64)

#     for t in range(nt):
#         # --- 1. wrap coordinates into the primary image --------------------
#         atoms_in_box = _wrap_into_box(pos_atom[t], L)
#         mols_in_box = _wrap_into_box(pos_mol[t], L)

#         # --- 2. build periodic KD-tree & query all molecules at once --------
#         tree = cKDTree(atoms_in_box, boxsize=L)

#         for i in range(nrad):
#             nbrs = tree.query_ball_point(mols_in_box, r=radius[i], workers=-1)
#             typed_nbrs = List(np.array(x, dtype=int) for x in nbrs)
#             dens[t] = neighbors_to_density(typed_nbrs, molecule_ids, atom_masses, V[i])

#     del atoms_in_box, mols_in_box, tree, nbrs
#     gc.collect()
#     return dens


# @nb.njit(parallel=True)
# def neighbors_to_density(nbrs, molecule_ids, atom_masses, V):

#     dens = np.empty(len(nbrs), dtype=np.float64)

#     for m_id in nb.prange(len(nbrs)):
#         idx = nbrs[m_id]
#         if idx.any():  # non-empty
#             foreign = molecule_ids[idx] != m_id  # mask out own atoms
#             mass_sum = atom_masses[idx][foreign].sum() if np.any(foreign) else 0.0
#             dens[m_id] = mass_sum / V
#         else:
#             dens[m_id] = 0

#     return dens


# import numpy as np
# from scipy.spatial import cKDTree
# import math


# def local_density_bincount(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, atoms_per_mol=44
# ):
#     """
#     Fully vectorised: no Python-level loop over molecules.
#     Returns ρ[t, i] for every (time-step, molecule).
#     """
#     nt, nmol, _ = pos_mol.shape
#     V = (4.0 / 3.0) * math.pi * radius**3
#     ρ = np.empty((nt, nmol), dtype=np.float64)

#     # True molecule-ID of each centre: 0..nmol-1 **only** if centres are sorted;
#     # else build a lookup once per step (shown inside loop below).
#     mol_id_of_centre = np.arange(nmol, dtype=molecule_ids.dtype)

#     for t in range(nt):
#         # --- wrap positions into [0,L) ---
#         atoms = np.mod(pos_atom[t], L)
#         centres = np.mod(pos_mol[t], L)

#         # --- k-d tree neighbour search (periodic) ---
#         tree = cKDTree(atoms, boxsize=L)
#         lists = tree.query_ball_point(centres, r=radius, workers=-1)

#         # ---------------------------------------------------------------
#         # 1. Flatten neighbour indices and build a parallel array that
#         #    stores *which centre requested* each neighbour.
#         # ---------------------------------------------------------------
#         lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#         owner = np.repeat(
#             np.arange(nmol, dtype=np.int32), lens
#         )  # (n_total_neighbours,)
#         neighbours = np.concatenate(lists)  # (n_total_neighbours,)

#         # ---------------------------------------------------------------
#         # 2. Mask out atoms that belong to the *same* molecule:
#         #    (molecule_ids == mol_id_of_centre[owner])
#         # ---------------------------------------------------------------
#         same = molecule_ids[neighbours] == mol_id_of_centre[owner]
#         weights = np.where(same, 0.0, atom_masses[neighbours])

#         # ---------------------------------------------------------------
#         # 3. Bin by `owner` to get total foreign mass per centre.
#         # ---------------------------------------------------------------
#         mass_per_centre = np.bincount(owner, weights=weights, minlength=nmol).astype(
#             np.float64
#         )

#         ρ[t] = mass_per_centre / V

#     return ρ


# import numpy as np
# from multiprocessing import Pool, cpu_count
# from functools import partial  # to curry fixed arguments


# # ---- your fast single-step kernel -----------------------------
# def density_one_step(t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     atoms = np.mod(pos_atom[t], L)
#     centres = np.mod(pos_mol[t], L)

#     from scipy.spatial import cKDTree  # import inside worker

#     tree = cKDTree(atoms, boxsize=L)
#     lists = tree.query_ball_point(centres, r=radius, workers=-1)

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     return mass_sum / V  # shape (nmol,)


# # ---- DRIVER ----------------------------------------------------
# def local_density_parallel_mp(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, chunk=50
# ):
#     nt = pos_mol.shape[0]
#     with Pool(min(cpu_count(), nt)) as pool:
#         step_func = partial(
#             density_one_step,
#             pos_mol=pos_mol,
#             pos_atom=pos_atom,
#             atom_masses=atom_masses,
#             molecule_ids=molecule_ids,
#             L=L,
#             radius=radius,
#         )
#         # Map in chunks of 50 steps to keep workers busy
#         densities = pool.map(step_func, range(nt), chunksize=chunk)
#     return np.vstack(densities)  # (nt, nmol)

# import numpy as np
# from joblib import Parallel, delayed
# import utils
# import sys, types, __main__
# from scipy.spatial import cKDTree  # import inside worker


# @nb.njit
# def wrap(x, L):

#     return np.mod(x, L)


# # ---- your fast single-step kernel -----------------------------
# def density_one_step(t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     atoms = np.mod(pos_atom[t], L)
#     centres = np.mod(pos_mol[t], L)

#     # atoms = wrap(pos_atom[t], L)
#     # centres = wrap(pos_mol[t], L)

#     # atoms = pos_atom[t]
#     # centres = pos_mol[t]

#     tree = cKDTree(atoms, boxsize=L)
#     lists = tree.query_ball_point(centres, r=radius, workers=-1)

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     dens = mass_sum / V
#     return dens


# # __main__.density_one_step = density_one_step


# def local_density_parallel_joblib(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, n_jobs=-1, batch_size=100
# ):
#     nt = pos_mol.shape[0]
#     # pos_mol = wrap(pos_mol, L)
#     # pos_atom = wrap(pos_atom, L)

#     def step(t):
#         return density_one_step(
#             t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)

# import numpy as np
# from joblib import Parallel, delayed
# import utils
# from scipy.spatial import cKDTree  # import inside worker


# # ---- your fast single-step kernel -----------------------------
# def density_one_step(t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     atoms = np.mod(pos_atom[t], L)
#     centres = np.mod(pos_mol[t], L)

#     tree = cKDTree(atoms, boxsize=L)
#     lists = tree.query_ball_point(centres, r=radius, workers=-1)

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     dens = mass_sum / V

#     return dens


# def _atomic_density(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, n_jobs=-1, batch_size=100
# ):
#     nt = pos_mol.shape[0]

#     def step(t):
#         return density_one_step(
#             t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)


# def atomic_density(df, radii, nt):

#     pos_atom = utils.get_pos(df, mode="atom")[:nt]
#     pos_mol = utils.get_pos(df)[:nt]
#     molecule_ids = df["atom"]["molecule_id"]
#     amu = df["AMU_TO_KG"]
#     atom_masses = df["atom"]["mass"] * amu
#     L = utils.get_L(df)

#     for radius in radii:

#         dens = _atomic_density(pos_mol, pos_atom, atom_masses, molecule_ids, L, radius)
#         df["molecule"][f"density_{radius}"] = dens

#     del pos_atom, pos_mol
#     gc.collect()

#     return None


# @nb.njit
# def _wrap_into_box(x, L):
#     """Return positions mapped into the range [0, L) for each axis."""
#     return np.mod(x, L)  # works for negative values too

# import numpy as np
# from joblib import Parallel, delayed
# import utils
# import sys, types, __main__
# from scipy.spatial import cKDTree  # import inside worker


# @nb.njit
# def wrap(x, L):

#     return np.mod(x, L)


# # ---- your fast single-step kernel -----------------------------
# def density_one_step(t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     atoms = wrap(pos_atom[t], L)
#     centres = wrap(pos_mol[t], L)

#     tree = cKDTree(atoms, boxsize=L)
#     lists = tree.query_ball_point(centres, r=radius, workersa=-1)

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     dens = mass_sum / V
#     return dens


# # __main__.density_one_step = density_one_step


# def local_density_parallel_joblib(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius, n_jobs=-1, batch_size=100
# ):
#     nt = pos_mol.shape[0]

#     def step(t):
#         return density_one_step(
#             t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radius
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)


# dens = local_density_parallel_joblib(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radius
# )


# import numpy as np
# from joblib import Parallel, delayed
# import utils
# import sys, types, __main__
# from scipy.spatial import cKDTree  # import inside worker


# @nb.njit
# def wrap(x, L):

#     return np.mod(x, L)


# # ---- your fast single-step kernel -----------------------------
# def density_one_step(t, pos_mol, pos_atom, atom_masses, molecule_ids, L, radii):
#     """Return densities for ONE time step (shape (nmol,))."""
#     nmol = pos_mol.shape[1]
#     # V = (4.0 / 3.0) * np.pi * radius**3
#     # --- wrap coords ---
#     # atoms = np.mod(pos_atom[t], L)
#     # centres = np.mod(pos_mol[t], L)

#     tree = cKDTree(pos_atom[t], boxsize=L)
#     # for radius in radii:
#     radius = radii
#     lists = tree.query_ball_point(pos_mol[t], r=radius, workers=-1)

#     lens = np.fromiter((len(x) for x in lists), dtype=np.int32)
#     owner = np.repeat(np.arange(nmol, dtype=np.int32), lens)
#     neigh = np.concatenate(lists)

#     same = molecule_ids[neigh] == owner  # pos_mol sorted ⇒ ID = owner
#     weights = np.where(same, 0.0, atom_masses[neigh])
#     mass_sum = np.bincount(owner, weights=weights, minlength=nmol)
#     dens = mass_sum
#     return dens


# # __main__.density_one_step = density_one_step


# def local_density_parallel_joblib(
#     pos_mol, pos_atom, atom_masses, molecule_ids, L, radii, n_jobs=-1, batch_size=100
# ):
#     nt = pos_mol.shape[0]
#     mol_wrap = wrap(pos_mol, L)
#     atom_wrap = wrap(pos_atom, L)

#     def step(t):
#         return density_one_step(
#             t, mol_wrap, atom_wrap, atom_masses, molecule_ids, L, radii
#         )

#     dens = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
#         delayed(step)(t) for t in range(nt)
#     )
#     return np.vstack(dens) / (1e-10**3)
