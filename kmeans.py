import numpy as np
import numpy.linalg as la
import numba as nb
import utils
import time

try:
    from sklearnex.cluster import KMeans as kmeans_sk
except:
    from sklearn.cluster import KMeans as kmeans_sk
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from density import density as Density
from joblib import Parallel, delayed
import os

higher_vars = [
    "abs(z)",
    "abs(vz)",
    "ke",
    "lt",
    "speed",
    "dz",
    "displacement",
    "pe",
]  # highter indicates gas
lower_vars = [
    "ld",
    "coordination",
    "density",
]  # lower indicates gas


@nb.njit
def init(data, k):
    idx = np.random.choice(data.shape[0], k, replace=False)

    if data.ndim == 1:
        for i in range(idx.shape[0]):
            for j in range(idx.shape[0]):
                if i == j:
                    continue

                ii = idx[i]
                jj = idx[j]

                if np.isclose(data[ii], data[jj]):
                    idx = np.random.choice(data.shape[0], k, replace=False)

        init_centroids = data[idx]

    else:
        for i in range(idx.shape[0]):
            for j in range(idx.shape[0]):
                if i == j:
                    continue
                ii = idx[i]
                jj = idx[j]

                if np.isclose(data[ii, :], data[jj, :]).any():
                    idx = np.random.choice(data.shape[0], k, replace=False)

        init_centroids = data[idx, :]

    return init_centroids


@nb.njit
def assign(data, centroids):
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int8)

    for i, xi in enumerate(data):
        minimum = 1e10
        centroid_idx = 0

        for j, centroid in enumerate(centroids):
            d = la.norm(xi - centroid) ** 2

            if d < minimum:
                minimum = d
                centroid_idx = j

        labels[i] = centroid_idx

    return labels


@nb.njit
def assign_external(data, centroids, external_norm):
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int8)

    for i, xi in enumerate(data):
        minimum = 1e10
        centroid_idx = 0

        for j, centroid in enumerate(centroids):
            a = (xi - centroid) / external_norm[j]
            d = la.norm(a)

            if d < minimum:
                minimum = d
                centroid_idx = j

        labels[i] = centroid_idx

    return labels


@nb.njit
def update(data, k, labels):
    if data.ndim == 1:
        centroids = np.zeros(k)

    else:
        centroids = np.zeros((k, data.shape[1]))

    centroid_counts = np.zeros(k)

    for idx, pair in enumerate(zip(data, labels)):
        xi, label = pair
        centroids[label] += xi
        centroid_counts[label] += 1

    for idx, count in enumerate(centroid_counts):
        centroids[idx] /= count

    return centroids


@nb.njit
def run(data, k, seed=0):
    np.random.seed(seed)
    centroids = init(data, k)

    i = 0
    while True:
        assignment = assign(data, centroids)
        prev_centroids = centroids.copy()
        centroids = update(data, k, assignment)

        if np.array_equal(prev_centroids, centroids):
            break

        i += 1

        if i % 100 == 0:
            print(f"Iteration {i}")

    return assignment, centroids


def run_sk(cluster_data, k, seed=0, nstart=10, tol=1e-12, return_model=False):
    res = kmeans_sk(
        n_clusters=k,
        verbose=0,
        init="random",
        n_init=nstart,
        tol=tol,
        max_iter=100000,
        random_state=seed,
    ).fit(cluster_data)
    centroids = res.cluster_centers_.copy()
    molecule_phase = res.predict(cluster_data)
    if return_model:
        return molecule_phase, centroids, res
    else:
        return molecule_phase, centroids


def swap(cluster_vars, centroids):
    global higher_vars
    global lower_vars
    mask = np.empty(len(cluster_vars), dtype=bool)
    cluster_vars = np.array(cluster_vars)

    for i, var in enumerate(cluster_vars):
        var_stripped = utils.strip_var(var)

        if (
            var_stripped in higher_vars
            or var_stripped.split("_")[0] == "dz"
            or var_stripped.split("_")[0] == "displacement"
        ):
            if (
                var_stripped == "z" or var_stripped == "vz" and "abs" not in var
            ):  # make sure abs(z)
                mask[i] = False
            else:
                # print(f'{var_stripped} higher gas norm set true')
                mask[i] = True

        elif (
            var_stripped in lower_vars
            or var_stripped.split("_")[0] == "coordination"
            or var_stripped.split("_")[0] == "density"
        ):
            # print(f'{var_stripped} higher gas norm set false')
            mask[i] = False

    higher_norm0 = la.norm(
        centroids[0][mask]
    )  # vars with higher gaseous norm for centroid 0
    lower_norm0 = la.norm(
        centroids[0][~mask]
    )  # vars with lower gaseous norm for centroid 0

    higher_norm1 = la.norm(
        centroids[1][mask]
    )  # vars with higher gaseous norm for centroid 1
    lower_norm1 = la.norm(
        centroids[1][~mask]
    )  # vars with lower gaseous norm for centroid 1

    # if centroid 0 has higher higher norm vars and
    # lower lower norm vars, we need to be in the gaseous state (1)
    # handle the case where we only have higher/lower norm variables

    higher_norm0 = centroids[0][mask]
    lower_norm0 = centroids[0][~mask]
    higher_norm1 = centroids[1][mask]
    lower_norm1 = centroids[1][~mask]
    swap_centroids = False

    if (higher_norm0 > higher_norm1).all() and (lower_norm0 < lower_norm1).all():
        swap_centroids = True

    elif (higher_norm0 > higher_norm1).any():
        print(f"Liquid state has higher {cluster_vars[mask]} norm")
        print(
            f"Liquid state: {centroids[0][mask]}, gasesous state: {centroids[1][mask]}"
        )
    elif (lower_norm0 < lower_norm1).any():
        print(f"Liquid state has lower {cluster_vars[~mask]} norm")
        print(
            f"Liquid state: {centroids[0][~mask]}, gasesous state: {centroids[1][~mask]}"
        )

    return swap_centroids


def prep_data(
    df,
    cluster_vars,
    start,
    nt,
    t=None,
    norm=None,
):

    cluster_vars = np.array(cluster_vars)
    nvars = len(cluster_vars)

    if t is not None:
        cluster_data = np.empty((df["nmolecule"], nvars))
        t += start

    else:
        cluster_data = np.empty((nt, df["nmolecule"], nvars))

    for i, var in enumerate(cluster_vars):

        feature = utils.parse(df, var)[start:]

        if t is not None:
            cluster_data[:, i] = utils.normalize_arr(feature[t], norm=norm)

        else:
            if isinstance(norm, np.ndarray):
                minmax = norm[i].copy()

                # if minmax.shape == (2,):
                #     if feature.max() > minmax[1]:
                #         minmax[1] = feature.max()
                #     if feature.min() < minmax[0]:
                #         minmax[0] = feature.min()

                # elif minmax.shape == (3,):
                #     minmax = minmax[:2]

                cluster_data[:, :, i] = (feature - minmax[0]) / (minmax[1] - minmax[0])
            elif isinstance(norm, str):
                cluster_data[:, :, i] = utils.normalize_arr(feature, norm=norm)

    if t is not None:
        if t <= start:
            raise ValueError(f"t must be greater than {start}")

    else:
        cluster_data = cluster_data.reshape(-1, nvars)

    return cluster_data


def get_centroids(df, start=0, actime=False):

    start = utils.get_lag(df, start, actime)

    cluster_vars = df["cluster_vars"]
    molecule_phase = df["molecule"]["phase"][start:]
    liquid = molecule_phase == 0
    gas = ~liquid

    liquid_centers = []
    gas_centers = []
    for var in cluster_vars:
        feat = utils.parse(df, var)[start:]
        liquid_center = np.mean(feat[liquid])
        gas_center = np.mean(feat[gas])
        liquid_centers.append(liquid_center)
        gas_centers.append(gas_center)

    return np.array([liquid_centers, gas_centers])


def get_centroids_manual(df, cluster_data, start):

    phase = df["molecule"]["phase"][start:].flatten()
    liquid = phase == 0
    gas = ~liquid
    liquid_centers = []
    gas_centers = []
    for i in range(cluster_data.shape[1]):
        feat = cluster_data[:, i]
        liquid_center = np.mean(feat[liquid])
        gas_center = np.mean(feat[gas])
        liquid_centers.append(liquid_center)
        gas_centers.append(gas_center)
    return np.array([liquid_centers, gas_centers])


def error(df, cluster_data, raw=False):
    err = {}
    start = df["start"]
    liquid = df["molecule"]["liquid"][start:].flatten()
    gas = df["molecule"]["gas"][start:].flatten()
    if raw:
        centroids = df["raw_centroids"]
    else:
        centroids = df["centroids"]

    nliquid = liquid.sum()
    ngas = gas.sum()
    nmolecule = df["nmolecule"]
    nt = df["nt"]
    nvars = cluster_data.shape[1]

    sse = SSE(cluster_data[liquid], centroids[0])
    sse += SSE(cluster_data[gas], centroids[1])

    err["sse"] = sse
    err["mse"] = sse / nmolecule / nt / nvars
    err["mse_liquid"] = SSE(cluster_data[liquid], centroids[0]) / nliquid / nvars
    err["mse_gas"] = SSE(cluster_data[gas], centroids[1]) / ngas / nvars

    return err


def classify_phase(
    df,
    cluster_vars,
    start=0,
    actime=False,
    t=None,
    k=2,
    to_print=False,
    mode="sk",
    seed=0,
    nstart=10,
    tol=1e-12,
    norm=None,
    external_centroids=None,  # normalized
    external_norm=None,
    return_iqr=False,
    return_std=False,
    return_raw_std=False,
    return_minmax=False,
    return_model=False,
    calc_distance=False,
    offset=False,
):
    if to_print:
        s = time.time()
        print("\nClassifying...", end="")

    df["cluster_vars"] = cluster_vars
    start = utils.get_lag(df, start, actime)
    df["start"] = start
    nt = df["nt"] - start
    nvars = len(cluster_vars)

    if t is not None:
        nt = 1

    cluster_data = prep_data(df, cluster_vars, start, nt, t, norm=norm)

    if external_centroids is not None:

        if external_norm is None:
            external_norm = np.ones((nvars))

        molecule_phase = assign_external(
            cluster_data, external_centroids, external_norm
        )

        centroids = external_centroids

    else:

        if mode == "self":

            molecule_phase, centroids = run(cluster_data, k, seed)

        else:
            if return_model:
                molecule_phase, centroids, model = run_sk(
                    cluster_data, k, seed, nstart, tol, return_model
                )
            else:
                molecule_phase, centroids = run_sk(cluster_data, k, seed, nstart, tol)

    molecule_phase = molecule_phase.reshape(nt, df["nmolecule"])
    swap_centroids = swap(cluster_vars, centroids)

    if swap_centroids:
        molecule_phase = 1 - molecule_phase
        tmp = centroids[1].copy()
        centroids[1] = centroids[0]
        centroids[0] = tmp

    atom_phase = np.empty((nt, df["natom"]), dtype=np.int8)
    metal = df["metal_mask"]
    non_metal = df["non_metal"]

    for i in range(nt):
        # apply molecule phases to atoms
        atom_phase[i][non_metal] = np.repeat(
            molecule_phase[i], df["natom_per_molecule"]
        )
        atom_phase[i][metal] = np.repeat(-1, df["nmetal"])

    if t is not None:
        molecule_phase = molecule_phase.flatten()
        atom_phase = atom_phase.flatten()

    else:
        molecule_phase = np.pad(
            molecule_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )
        atom_phase = np.pad(
            atom_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )

    df["molecule"]["phase"] = molecule_phase
    df["atom"]["phase"] = atom_phase
    df["centroids"] = centroids
    liquid = molecule_phase == 0
    gas = molecule_phase == 1
    df["molecule"]["nliquid"] = np.sum(liquid, axis=1)
    df["molecule"]["ngas"] = np.sum(gas == 1, axis=1)
    df["molecule"]["liquid"] = liquid
    df["molecule"]["gas"] = gas

    err = error(df, cluster_data)
    raw_centroids = get_centroids(df, start=start, actime=actime)
    df["raw_centroids"] = raw_centroids
    cluster_data_raw = prep_data(df, cluster_vars, start=start, nt=nt, t=t, norm=None)
    raw_err = error(df, cluster_data_raw, raw=True)

    df["err"] = err
    for key, value in raw_err.items():
        df["err"]["raw_" + key] = value

    if calc_distance:

        dist_err = average_distance(df, cluster_data)
        raw_dist_err = average_distance(df, cluster_data, raw=True)

        for key, value in dist_err.items():
            df["err"][key] = value

        for key, value in raw_dist_err.items():
            df["err"]["raw_" + key] = value

    if (liquid.sum() > 0) and (gas.sum() > 0):
        try:
            df["sil"] = silhouette_score(
                cluster_data,
                molecule_phase[start:].flatten(),
                n_jobs=None,
                sample_size=10000,
                random_state=0,
            )
        except:
            try:
                df["sil"] = silhouette_score(
                    cluster_data,
                    molecule_phase[start:].flatten(),
                    n_jobs=None,
                    sample_size=10000,
                    random_state=1,
                )
            except:
                df["sil"] = np.nan
    else:
        df["sil"] = np.nan

    try:
        utils.generate_switch_info(df, start=start, actime=actime, offset=offset)
    except Exception as e:
        print("Error generating switch info")
        print(e, end="\n")

    to_return = [molecule_phase, atom_phase, centroids]

    if return_iqr:

        iqr = class_iqrs(df, cluster_vars, start)
        to_return.append(iqr)

    if return_raw_std:

        std = class_stds(df, cluster_vars, start)
        to_return.append(std)

    if return_std:

        std = class_stds(df, cluster_vars, start, raw=False)
        to_return.append(std)

    if return_minmax:

        minmax = np.zeros((nvars, 2))

        for i in range(nvars):

            x = utils.parse(df, cluster_vars[i])[start:]
            minmax[i, 0] = x.min()
            minmax[i, 1] = x.max()

        to_return.append(minmax)

    if return_model:
        to_return.append(model)

    if to_print:
        e = time.time()
        dt = e - s
        print(f" Done in {dt:.2f}s")

    return to_return


def classify_phase_manual(
    dfs,
    cluster_data,
    cluster_vars,
    nt,
    temps,
    start=0,
    actime=False,
    t=None,
    k=2,
    to_print=False,
    mode="sk",
    seed=0,
    nstart=10,
    tol=1e-12,
    norm=None,
    external_centroids=None,
    external_norm=None,
    return_iqr=False,
    return_std=False,
    return_model=False,
):
    from copy import deepcopy

    if to_print:
        print("\nClassifying...\n")

    results = {}
    dfs["results"] = results
    ref_df = dfs[temps[0]]
    ref_df["cluster_vars"] = cluster_vars

    results["cluster_vars"] = cluster_vars
    nvars = len(cluster_vars)
    ntemps = len(temps)
    start = utils.get_lag(ref_df, start, actime)
    results["start"] = start
    results["nt"] = nt
    results["molecule"] = {}
    results["atom"] = {}
    nmol = ref_df["nmolecule"] * ntemps
    results["nmolecule"] = nmol
    xs = []
    ys = []
    zs = []
    for temp in temps:
        x = dfs[temp]["molecule"]["x"][:nt]
        y = dfs[temp]["molecule"]["y"][:nt]
        z = dfs[temp]["molecule"]["z"][:nt]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.concatenate(xs, axis=1)
    ys = np.concatenate(ys, axis=1)
    zs = np.concatenate(zs, axis=1)
    # xs = xs.reshape(nt, nmol)
    # ys = ys.reshape(nt, nmol)
    # zs = zs.reshape(nt, nmol)

    results["molecule"]["x"] = xs
    results["molecule"]["y"] = ys
    results["molecule"]["z"] = zs

    if external_centroids is not None:

        if external_norm is None:
            external_norm = np.ones((nvars))

        molecule_phase = assign_external(
            cluster_data, external_centroids, external_norm
        )

        centroids = external_centroids

    else:

        if mode == "self":

            molecule_phase, centroids = run(cluster_data, k, seed)

        else:
            if return_model:
                molecule_phase, centroids, model = run_sk(
                    cluster_data, k, seed, nstart, tol, return_model
                )
            else:
                molecule_phase, centroids = run_sk(cluster_data, k, seed, nstart, tol)
    molecule_phase = molecule_phase.reshape(nt, nmol)
    swap_centroids = swap(cluster_vars, centroids)

    if swap_centroids:
        molecule_phase = 1 - molecule_phase
        tmp = centroids[1].copy()
        centroids[1] = centroids[0]
        centroids[0] = tmp

    atom_phase = np.empty((nt, ref_df["natom"] * ntemps), dtype=np.int8)
    metal = ref_df["metal_mask"]
    non_metal = ref_df["non_metal"]
    metal = np.repeat(metal, ntemps)
    non_metal = np.repeat(non_metal, ntemps)

    for i in range(nt):
        # apply molecule phases to atoms
        atom_phase[i][non_metal] = np.repeat(
            molecule_phase[i], ref_df["natom_per_molecule"]
        )
        atom_phase[i][metal] = np.repeat(-1, ref_df["nmetal"])

    if t is not None:
        molecule_phase = molecule_phase.flatten()
        atom_phase = atom_phase.flatten()

    else:
        molecule_phase = np.pad(
            molecule_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )
        atom_phase = np.pad(
            atom_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )

    results["molecule"]["phase"] = molecule_phase
    results["atom"]["phase"] = atom_phase
    results["centroids"] = centroids
    liquid = molecule_phase == 0
    gas = molecule_phase == 1
    results["nliquid"] = np.sum(liquid, axis=1)
    results["ngas"] = np.sum(gas == 1, axis=1)
    results["liquid"] = liquid
    results["gas"] = gas

    err = error(results, cluster_data)

    if to_print:
        print("Classification done\n")

    # utils.generate_switch_info(results, start=start, actime=actime)
    to_return = [molecule_phase, atom_phase, centroids, err]

    # if return_iqr:

    #     iqr = class_iqrs_manual(results, cluster_vars, start)
    #     to_return.append(iqr)

    # if return_std:

    #     std = class_stds_manual(results, cluster_vars, start)
    #     to_return.append(std)

    # if return_model:
    #     to_return.append(model)

    return to_return


def prep_data_manual(
    dfs,
    temps,
    cluster_vars,
    nt,
    t=None,
    norm=None,
):

    cluster_vars = np.array(cluster_vars)
    nvars = len(cluster_vars)
    ntemps = len(temps)
    nmolecule = dfs[temps[0]]["nmolecule"]
    cluster_data = np.empty((nt, nmolecule * ntemps, nvars))

    for j, temp in enumerate(temps):
        df = dfs[temp]

        for i, var in enumerate(cluster_vars):

            feature = utils.parse(df, var)[:nt]
            # print(nt * j, nt * (j + 1), nmolecule * j, nmolecule * (j + 1))
            cluster_data[:, nmolecule * j : nmolecule * (j + 1), i] = (
                utils.normalize_arr(feature, norm=norm)
            )
    cluster_data = cluster_data.reshape(-1, nvars)

    return cluster_data


def SSE(data, centroid):
    return np.sum(la.norm(data - centroid) ** 2)  # fastest for large N
    # return np.square((data - centroid)).sum() slower
    # return np.sum((data - centroid) ** 2) slower


def class_stds(df, cluster_vars, start, raw=True):
    nvars = len(cluster_vars)
    liquid = df["molecule"]["liquid"][start:]
    gas = df["molecule"]["gas"][start:]
    std = np.zeros((2, nvars))
    for i in range(nvars):

        if raw:
            x = utils.parse(df, cluster_vars[i])[start:]
        else:
            x = utils.parse(df, "norm(" + cluster_vars[i] + ")")[start:]
        x_gas = x[gas]
        x_liq = x[liquid]

        std[0, i] = np.std(x_liq)
        std[1, i] = np.std(x_gas)
    return std


def feature_rmse(df, cluster_vars, start, norm=None):

    if norm is not None:
        centroids = df["centroids"]
    else:
        centroids = df["raw_centroids"]
    nvars = len(cluster_vars)
    liquid = df["molecule"]["liquid"][start:]
    gas = df["molecule"]["gas"][start:]
    rmse = np.zeros((3, nvars))
    nt = df["nt"] - start
    nmol = df["nmolecule"]
    for i in range(nvars):
        x = utils.parse(df, cluster_vars[i])[start:]
        if norm is not None:
            x = utils.normalize_arr(x, norm=norm)
        x_gas = x[gas]
        x_liq = x[liquid]
        n_liq = x_liq.flatten().shape[0]
        n_gas = x_gas.flatten().shape[0]

        se_liq = (x_liq - centroids[0, i]) ** 2
        se_gas = (x_gas - centroids[1, i]) ** 2
        mse_liq = np.mean(se_liq)
        mse_gas = np.mean(se_gas)

        sse_tot = np.sum(se_liq) + np.sum(se_gas)
        mse_tot = sse_tot / nmol / nt
        mse_tot = sse_tot / (se_liq.flatten().shape[0] + se_gas.flatten().shape[0])

        rmse[0, i] = np.sqrt(mse_liq)
        rmse[1, i] = np.sqrt(mse_gas)
        rmse[2, i] = np.sqrt(mse_tot)

    return rmse


def average_distance(df, cluster_data, raw=False):
    err = {}
    start = df["start"]
    liquid = df["molecule"]["liquid"][start:].flatten()
    gas = df["molecule"]["gas"][start:].flatten()
    if raw:
        centroids = df["raw_centroids"]
    else:
        centroids = df["centroids"]

    avg_dist = np.zeros(3)

    gas_data = cluster_data[gas]
    liq_data = cluster_data[liquid]
    gas_disp = la.norm(gas_data - centroids[1], axis=1)
    liq_disp = la.norm(liq_data - centroids[0], axis=1)
    avg_dist[0] = np.mean(liq_disp)
    avg_dist[1] = np.mean(gas_disp)
    avg_dist[2] = np.mean(np.concatenate((liq_disp, gas_disp)))

    err["avg_dist_liq"] = avg_dist[0]
    err["avg_dist_gas"] = avg_dist[1]
    err["avg_dist"] = avg_dist[2]

    return err


def class_stds_manual(df, cluster_data, start):
    nvars = cluster_data.shape[1]
    liquid = df["molecule"]["liquid"][start:].flatten()
    gas = df["molecule"]["gas"][start:].flatten()
    std = np.zeros((2, nvars))
    for i in range(nvars):
        x = cluster_data[:, i]
        x_gas = x[gas]
        x_liq = x[liquid]

        std[0, i] = np.std(x_liq)
        std[1, i] = np.std(x_gas)
    return std


def class_iqrs(df, cluster_vars, start):
    nvars = len(cluster_vars)
    liquid = df["molecule"]["liquid"][start:]
    gas = df["molecule"]["gas"][start:]
    iqr = np.zeros((2, nvars))
    for i in range(nvars):
        x = utils.parse(df, cluster_vars[i])[start:]
        x_gas = x[gas]
        x_liq = x[liquid]
        q25, q75 = np.percentile(x_liq, [25, 75])
        iqr[0, i] = q75 - q25
        q25, q75 = np.percentile(x_gas, [25, 75])
        iqr[1, i] = q75 - q25
    return iqr


def class_iqrs_manual(df, cluster_data, start):
    nvars = cluster_data.shape[1]
    liquid = df["molecule"]["liquid"][start:].flatten()
    gas = df["molecule"]["gas"][start:].flatten()
    iqr = np.zeros((2, nvars))
    for i in range(nvars):
        x = cluster_data[:, i]
        x_gas = x[gas]
        x_liq = x[liquid]
        q25, q75 = np.percentile(x_liq, [25, 75])
        iqr[0, i] = q75 - q25
        q25, q75 = np.percentile(x_gas, [25, 75])
        iqr[1, i] = q75 - q25
    return iqr


def density(
    df,
    bin_width=2,
    mode="atom",
    time_avg=True,
    absval=True,
    center=True,
    actime=False,
    auto_range=True,
    std=True,
    norm="mass",
    start=0,
):
    liq_density, liq_zbin, liq_err = Density(
        df,
        bin_width=bin_width,
        mode=mode,
        phase_mask=0,
        time_avg=time_avg,
        absval=absval,
        center=center,
        actime=actime,
        auto_range=auto_range,
        std=std,
        norm=norm,
        start=start,
    )

    gas_density, gas_zbin, gas_err = Density(
        df,
        bin_width=bin_width,
        mode=mode,
        phase_mask=1,
        time_avg=time_avg,
        absval=absval,
        center=center,
        actime=actime,
        auto_range=auto_range,
        std=std,
        norm=norm,
        start=start,
    )

    return liq_density, liq_zbin, liq_err, gas_density, gas_zbin, gas_err


def plot_density(
    liq_density,
    liq_zbin,
    liq_err,
    gas_density,
    gas_zbin,
    gas_err,
    ax=None,
    figsize=(4, 3),
    title="",
    std=False,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        # ax.set_xlabel("Z (Å)")
        # ax.set_ylabel(r"$\rho \ (kg \cdot m^{-3})$")
        ax.set_xlabel(r"$Z \ (\text{Å})$")
        ax.set_ylabel(r"$\rho \ (\mathrm{g cm^{-3}})$")
        ax.set_title(title)

    ax.plot(gas_zbin, gas_density, color="darkred", label="$kmeans_{gas}$", zorder=10)
    ax.plot(liq_zbin, liq_density, color="cyan", label="$kmeans_{liq}$", linestyle="-")

    if gas_err is not None and std:

        ax.fill_between(
            liq_zbin,
            liq_density - liq_err,
            liq_density + liq_err,
            alpha=0.4,
            color="green",
        )
        ax.fill_between(
            gas_zbin,
            gas_density - gas_err,
            gas_density + gas_err,
            alpha=0.4,
            color="yellow",
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax


def fit_density(kmeans_density, temp, window=11, threshold=0.5):
    from scipy import signal

    liq_density, liq_zbin, liq_err, gas_density, gas_zbin, gas_err = kmeans_density

    liq_threshold = 2000 / temp
    gas_threshold = 0.5
    if temp == 400:
        liq_threshold = 1
    elif temp > 400:
        liq_threshold = 0.5

    try:
        x1 = liq_zbin
        y1 = liq_density

        if len(y1) < window + 1:
            y1_smooth = y1
        else:
            y1_smooth = signal.savgol_filter(y1, window_length=7, polyorder=1)

        dy1 = np.gradient(y1_smooth, x1)
        linear_regions = np.abs(dy1) < liq_threshold
        linear_region_indices = np.where(linear_regions)[0]
        diff_indices = np.diff(linear_region_indices)
        split_points = np.where(diff_indices > 1)[0] + 1

        segments = np.split(linear_region_indices, split_points)
        rho_l = np.mean(y1[segments[0]])  # take mean of first linear region
        if rho_l < 100:  # for close to fully vaporized, no linear regime really
            rho_l = np.mean(y1[~segments[0]])
    except:
        rho_l = 0

    try:
        x2 = gas_zbin
        y2 = gas_density

        if len(y2) < window + 1:
            y2_smooth = y2
        else:
            y2_smooth = signal.savgol_filter(y2, window_length=4, polyorder=2)
        dy2 = np.gradient(y2_smooth, x2)
        linear_regions = np.abs(dy2) < gas_threshold
        linear_region_indices = np.where(linear_regions)[0]
        diff_indices = np.diff(linear_region_indices)
        split_points = np.where(diff_indices > 1)[0] + 1

        segments = np.split(linear_region_indices, split_points)

        if len(y2[segments[-1]]) > 0:
            rho_g = np.mean(y2[segments[-1]])  # take mean of last linear region
            if (x2 > 100).any():
                mask = np.where(x2 > 100)[0]
                rho_g = np.mean(y2[mask])
            else:
                rho_g = 0
        else:
            rho_g = 0

    except:
        rho_g = 0

    return rho_l, rho_g


def plot_feature_distributions(
    df, temp, cluster_vars, mode="molecule", figsize=None, nbins=None, title=""
):

    nvars = len(cluster_vars)
    start = utils.get_lag(df[temp], actime=True)
    Nt = df[temp]["nt"] - start
    if figsize is None:
        figsize = (4, 2 * nvars)
    fig, ax = plt.subplots(
        nrows=nvars, ncols=1, figsize=figsize, dpi=200, layout="constrained"
    )
    x = np.zeros((nvars, 2))
    x_gas = np.zeros((nvars, 2))
    x_liquid = np.zeros((nvars, 2))
    phase = df[temp][mode]["phase"][start:]
    gas = phase == 1
    liquid = ~gas
    if nvars == 1:
        ax = [ax]

    for j in range(nvars):
        try:
            x = utils.parse(df[temp], cluster_vars[j], mode=mode)[start:]
        except:

            raise KeyError(f"{cluster_vars[j]} not in {mode} data. Exiting...")

        if j == 0:
            gas_label = "gas"
            liquid_label = "liquid"
        else:
            gas_label = None
            liquid_label = None

        x_gas = x[gas]
        x_liq = x[liquid]
        bins = nbins[j] if nbins is not None else 10

        counts, bin_edges = np.histogram(x, bins=bins)
        N = counts.sum()

        counts, edges = np.histogram(x_liq, bins=bin_edges)
        counts = counts.astype(float) / N
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax[j].plot(center, counts, color="navy", label=liquid_label)

        counts, edges = np.histogram(x_gas, bins=bin_edges)
        counts = counts.astype(float) / N
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax[j].plot(center, counts, color="darkred", label=gas_label)

        ax[j].minorticks_on()
        ax[j].grid()
        ax[j].set_ylabel("Probability")

    feature_labels = [utils.parse_label(var) for var in cluster_vars]

    for j in range(nvars):
        feature = feature_labels[j]
        label = utils.parse_ylabel(cluster_vars[j])
        ax[j].set_xlabel(label)
        ax[j].set_title(feature_labels[j])

    fig.legend(bbox_to_anchor=(1.35, 0.5), loc="outside right")
    fig.suptitle(title, x=0.6)

    return fig, ax


def plot_feature_means(df, temps, cluster_vars, figsize=(6, 3)):

    nvars = len(cluster_vars)
    ntemps = len(temps)

    fig, ax = plt.subplots(
        nrows=nvars, ncols=1, figsize=figsize, dpi=200, layout="constrained"
    )
    x = np.zeros((ntemps, nvars))
    x_gas = np.zeros((ntemps, nvars))
    x_liquid = np.zeros((ntemps, nvars))

    for i, temp in enumerate(temps):

        start = utils.get_lag(df[temp], actime=True)

        for j in range(nvars):
            x[i, j] = utils.parse(df[temp], cluster_vars[j])[start:].mean()

        phase = df[temp]["molecule"]["phase"][start:]
        gas = phase == 1
        liquid = ~gas

        if gas.sum() == 0:
            for j in range(nvars):
                x_gas[i, j] = np.nan

        else:
            for j in range(nvars):
                x_gas[i, j] = utils.parse(df[temp], cluster_vars[j])[start:][gas].mean()

        if liquid.sum() == 0:
            for j in range(nvars):
                x_liquid[i, j] = np.nan

        else:
            for j in range(nvars):
                x_liquid[i, j] = utils.parse(df[temp], cluster_vars[j])[start:][
                    liquid
                ].mean()

    feature_labels = [utils.parse_label(var) for var in cluster_vars]

    for j in range(nvars):
        if j == 0:
            gas_label = "gas"
            liquid_label = "liquid"
        else:
            gas_label = None
            liquid_label = None

        ax[j].plot(temps, x[:, j], color="tab:purple", label=None)
        ax[j].set_title(feature_labels[j])

        ax[j].plot(temps, x_gas[:, j], color="tab:red", label=gas_label)
        ax[j].plot(temps, x_liquid[:, j], color="tab:blue", label=liquid_label)

    ax[-1].set_xlabel("T (K)")
    # ax[-1, 1].set_xlabel('T (K)')
    # ax[-1, 2].set_xlabel('T (K)')

    for j in range(nvars):
        feature = feature_labels[j]
        label = utils.parse_ylabel(cluster_vars[j])
        ax[j].set_ylabel(label)

    fig.legend(bbox_to_anchor=(1.35, 0.5), loc="outside right")
    fig.suptitle("Feature Mean Values vs Temperature")
    return fig, ax


# this is somehow slower than extra square root/square in la.norm
# def sse(data, centroid):
# return np.sum(np.sum((data - centroid) ** 2, axis=1))


def get_err_sweep(df, radii=None, lags=None, norm=True, start=0):

    if radii is None and lags is None:
        print("No radii or lags")
        return None

    if radii is not None and lags is not None:
        print("Select ONE of radii or lags, else use get_rmse_grid()")
        return None

    if radii is not None:
        feats = [[f"coordination_{radius}"] for radius in radii]
    else:
        feats = [[f"displacement_{lag}"] for lag in lags]

    rmse = []
    rmse_gas = []
    rmse_liq = []

    dist = []
    dist_gas = []
    dist_liq = []

    for feat in feats:

        molecule_phase, atom_phase, centroids, err = classify_phase(
            df,
            feat,
            norm="minmax",
            actime=False,
            start=start,
        )
        if norm:
            d_liq, d_gas, d = average_distance(df, feat, start, norm="minmax")
        else:
            err = df["err"]["raw"]
            d_liq, d_gas, d = average_distance(df, feat, start, norm=None)

        r_gas = np.sqrt(err["mse_gas"])
        r_liq = np.sqrt(err["mse_liquid"])
        r = np.sqrt(err["mse"])

        rmse.append(r)
        rmse_gas.append(r_gas)
        rmse_liq.append(r_liq)

        dist.append(d)
        dist_gas.append(d_gas)
        dist_liq.append(d_liq)

    rmse = np.array(rmse)
    rmse_gas = np.array(rmse_gas)
    rmse_liq = np.array(rmse_liq)

    dist = np.array(dist)
    dist_gas = np.array(dist_gas)
    dist_liq = np.array(dist_liq)

    rmse_dict = {"rmse": rmse, "rmse_liquid": rmse_liq, "rmse_gas": rmse_gas}
    dist_dict = {"dist": dist, "dist_liquid": dist_liq, "dist_gas": dist_gas}

    if norm:
        df["rmse"] = rmse_dict
        df["dist"] = dist_dict
    else:
        df["raw_rmse"] = rmse_dict
        df["raw_dist"] = dist_dict

    return rmse_dict, dist_dict


def get_err_grid(
    df, radii, lags, start=0, plot=False, metric="rmse", raw=False, pe=False, dpi=100
):

    errs = []
    # to_return = []

    if plot:
        cmap = plt.get_cmap("coolwarm", len(lags))
        legend_loc = (1350 / 1300, 0.40)
        fig, ax = plt.subplots(dpi=dpi)
        xlabel = "Radius (Å)"
        if metric == "rmse":
            title = f"{df['molecule_name']} {df['temp']}K RMSE"
            ylabel = "RMSE"
        elif metric == "dist":
            title = f"{df['molecule_name']} {df['temp']}K Mean Distance to Centroid"
            ylabel = "Distance"
        if pe:
            title += " - with PE"

    for i, lag in enumerate(lags):
        err = []

        for rad in radii:
            cluster_vars = [f"displacement_{lag}", f"density_{rad}"]
            if pe:
                cluster_vars.append("pe")

            molecule_phase, atom_phase, centroids = classify_phase(
                df,
                cluster_vars,
                norm="minmax",
                actime=False,
                start=start,
                calc_distance=True if (metric == "dist" or metric == "both") else False,
            )
            if raw:
                mse = df["err"]["raw_mse"]
            else:
                mse = df["err"]["mse"]

            e = np.sqrt(mse)

            if metric == "dist":
                if raw:
                    e = df["err"]["raw_avg_dist"]
                else:
                    e = df["err"]["avg_dist"]

            err.append(e)

        errs.append(err)

        if plot:
            x = np.array(radii)
            y = err
            ax.plot(x, y, label=f"{lag}", color=cmap(i), marker="o", markersize=4)

    if plot:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        ax.minorticks_on()
        legend = ax.legend(
            fancybox=False, edgecolor="black", fontsize=10, loc=legend_loc
        )
        legend.get_frame().set_linewidth(0.5)
        legend.set_title(
            "Lag",
            prop={"size": 12},
        )
        return fig, ax, errs
    else:
        return errs


def plot_feature_distributions_one_axis(
    df,
    cluster_vars,
    mode="molecule",
    figsize=None,
    nbins=None,
    title="",
    actime=False,
    minmax=True,
):

    nvars = len(cluster_vars)
    start = utils.get_lag(df, actime=actime)
    Nt = df["nt"] - start
    if figsize is None:
        figsize = (4, 2 * nvars)
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=figsize, dpi=200, layout="constrained"
    )

    phase = df[mode]["phase"][start:]
    gas = phase == 1
    liquid = ~gas
    ngas = gas.sum()
    nliquid = liquid.sum()
    if nvars == 1:
        ax = [ax]
    if isinstance(nbins, int):
        nbins = [nbins] * nvars

    colors = ["tab:blue", "tab:orange", "green"]
    colors = ["0C5DA5", "00B945", "FF9500", "FF2C00", "845B97", "474747", "9e9e9e"]
    feature_labels = [utils.parse_label(var) for var in cluster_vars]
    for j in range(nvars):
        try:
            x = utils.parse(df, cluster_vars[j], mode=mode)[start:]
        except:

            raise KeyError(f"{cluster_vars[j]} not in {mode} data. Exiting...")

        if j == 0:
            gas_label = "gas"
            liquid_label = "liquid"
        else:
            gas_label = None
            liquid_label = None
        if minmax:
            x = utils.normalize_arr(x, norm="minmax")

        x_gas = x[gas]
        x_liq = x[liquid]
        bins = nbins[j] if nbins is not None else 20

        counts, bin_edges = np.histogram(x, bins=bins)
        N = counts.sum()

        counts, edges = np.histogram(x_liq, bins=bin_edges)
        counts = counts.astype(float) / nliquid
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax.plot(
            center,
            counts,
            color=colors[j],
            label=feature_labels[j],
            linewidth=2,
            linestyle="--",
        )

        counts, edges = np.histogram(x_gas, bins=bin_edges)
        counts = counts.astype(float) / ngas
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax.plot(
            center,
            counts,
            color=colors[j],
            linestyle="-",
            linewidth=2,
            # dashes = (5,5),
        )

    # for j in range(nvars):
    #     feature = feature_labels[j]
    #     if not minmax:
    #         label = utils.parse_ylabel(cluster_vars[j])
    #         ax[j].set_xlabel(label)
    #     ax[j].set_title(feature_labels[j])
    ax.minorticks_on()
    ax.grid()
    ax.set_ylim(0, 0.15)
    ax.set_ylabel("Probability")
    # fig.legend(bbox_to_anchor=(1.15, 0.5), loc="outside right")
    ax.legend()
    fig.suptitle(title, x=0.6)

    return fig, ax


def plot_var_distribution(
    df,
    var,
    mode="molecule",
    figsize=None,
    nbins=None,
    title="",
    actime=False,
    minmax=True,
    all_counts=False,
    start=0,
    ax=None,
):
    from matplotlib.ticker import FormatStrFormatter

    start = utils.get_lag(df, actime=actime, start=start)
    Nt = df["nt"] - start

    if ax is None:
        if figsize is None:
            figsize = (4, 2)
        fig, ax = plt.subplots(figsize=figsize, dpi=400)

    phase = df[mode]["phase"][start:]
    gas = phase == 1
    liquid = ~gas

    N = gas.sum() + liquid.sum()
    if all_counts:
        ngas = N
        nliq = N
    else:
        ngas = gas.sum()
        nliq = liquid.sum()

    try:
        x = utils.parse(df, var, mode=mode)[start:]
    except:

        raise KeyError(f"{var} not in {mode} data. Exiting...")

    if minmax:
        x = utils.normalize_arr(x, norm="minmax")

    x_gas = x[gas]
    x_liq = x[liquid]
    if nbins is None:
        bins = 20
    else:
        bins = nbins

    counts, bin_edges = np.histogram(x, bins=bins)
    N = counts.sum()

    counts, edges = np.histogram(x_liq, bins=bin_edges)
    counts = counts.astype(float) / nliq
    counts[counts == 0] = np.nan
    center = (edges[1:] + edges[:-1]) / 2
    ax.plot(counts, center, color="navy")  # , linewidth=2)

    counts, edges = np.histogram(x_gas, bins=bin_edges)
    counts = counts.astype(float) / ngas
    counts[counts == 0] = np.nan
    center = (edges[1:] + edges[:-1]) / 2
    ax.plot(counts, center, color="darkred")  # , linewidth=2)

    ax.minorticks_on()
    ax.grid()
    # ax[j].set_ylabel("Probability", fontsize=14)
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    feature_label = utils.parse_ylabel(var)
    # ax.set_ylabel(feature_label, rotation=270, labelpad=12)
    ax.set_ylim(x.min(), x.max())

    return None


def one_sweep(df, radius, lag, distance):

    cluster_vars = [f"density_{radius}", f"displacement_{lag}", "pe"]

    _ = classify_phase(df, cluster_vars, norm="minmax", calc_distance=distance)
    sil = df["sil"]

    r = np.sqrt(df["err"]["mse"])
    r_liq = np.sqrt(df["err"]["mse_liquid"])
    r_gas = np.sqrt(df["err"]["mse_gas"])

    if distance:
        d = df["err"]["avg_dist"]
        d_liq = df["err"]["avg_dist_liq"]
        d_gas = df["err"]["avg_dist_gas"]

        return np.array(
            [
                sil,
                r,
                r_liq,
                r_gas,
                d,
                d_liq,
                d_gas,
            ]
        )

    else:
        return np.array([sil, r, r_liq, r_gas])


def sweep(df, radii, lags, distance=False):

    nl = len(lags)
    nr = len(radii)
    ncpus = os.cpu_count()
    njobs = min(nr, ncpus)

    if distance:
        data_out = np.empty((nl, nr, 7), dtype=np.float64)
    else:
        data_out = np.empty((nl, nr, 4), dtype=np.float64)

    for il, lag in enumerate(lags):

        results = np.array(
            [
                data
                for data in Parallel(n_jobs=njobs)(
                    delayed(one_sweep)(df, radius, lag, distance) for radius in radii
                )
            ]
        )

        data_out[il, :, :] = results

    return data_out


def feat_optim(df, radii, lags, distance=False):

    results = sweep(df, radii, lags, distance)

    ret = {
        "sil": None,
        "rmse": None,
        "rmse_liq": None,
        "rmse_gas": None,
        "d": None,
        "d_liq": None,
        "d_gas": None,
    }
    keys = list(ret.keys())

    for k in range(results.shape[-1]):

        ret[keys[k]] = results[:, :, k].T  # (rad, lag) format

    return ret


def plot_density_fit(kmeans_density, temp, window=11, threshold=0.5):
    from scipy import signal

    fig, axs = plt.subplots(ncols=2, dpi=400)
    ax = axs[0]
    ax1 = axs[1]
    liq_density, liq_zbin, liq_err, gas_density, gas_zbin, gas_err = kmeans_density

    liq_threshold = 2000 / temp
    gas_threshold = 0.5
    if temp == 400:
        liq_threshold = 1
    elif temp > 400:
        liq_threshold = 0.5

    x1 = liq_zbin
    y1 = liq_density

    ax.plot(x1, y1 / 1000, color="darkred", label="data")

    if len(y1) < window + 1:
        y1_smooth = y1
    else:
        y1_smooth = signal.savgol_filter(y1, window_length=7, polyorder=1)

    ax.plot(x1, y1_smooth / 1000, color="blue", label="smooth")

    try:
        dy1 = np.gradient(y1_smooth, x1)
        linear_regions = np.abs(dy1) < liq_threshold
        linear_region_indices = np.where(linear_regions)[0]
        diff_indices = np.diff(linear_region_indices)
        split_points = np.where(diff_indices > 1)[0] + 1

        segments = np.split(linear_region_indices, split_points)
        rho_l = np.mean(y1[segments[0]])  # take mean of first linear region
        if rho_l < 100:  # for close to fully vaporized, no linear regime really
            rho_l = np.mean(y1[~segments[0]])

        for ii, seg in enumerate(segments):
            if len(segments) > 1:
                if ii == len(segments) - 1:
                    break
            ax.plot(
                x1[seg],
                y1[seg] / 1000,
                linestyle="--",
                marker="o",
                color="black",
                markersize=4,
            )
    except Exception as e:
        rho_l = 0
        print("error fitting kmeans liq density")
        print(e, end="\n")

    ax.set_xlabel("Z (Å)")
    ax.set_ylabel(r"$\rho \ (\mathrm{g \ cm^{-3}})$")

    x2 = gas_zbin
    y2 = gas_density
    ax1.plot(x2, y2 / 1000, color="darkred")

    if len(y2) < window + 1:
        y2_smooth = y2
    else:
        y2_smooth = signal.savgol_filter(y2, window_length=4, polyorder=2)
    ax1.plot(x2, y2_smooth / 1000, color="blue", linestyle="--")

    try:
        dy2 = np.gradient(y2_smooth, x2)
        linear_regions = np.abs(dy2) < gas_threshold
        linear_region_indices = np.where(linear_regions)[0]
        diff_indices = np.diff(linear_region_indices)
        split_points = np.where(diff_indices > 1)[0] + 1

        segments = np.split(linear_region_indices, split_points)
        # print(x2[segments[-1]], y2[segments[-1]])

        ax1.plot(
            x2[segments[-1]],
            y2[segments[-1]] / 1000,
            linestyle="--",
            marker="o",
            color="black",
            markersize=4,
        )

        if len(y2[segments[-1]]) > 0:
            rho_g = np.mean(y2[segments[-1]])  # take mean of last linear region
            if (x2 > 100).any():
                mask = np.where(x2 > 100)[0]
                rho_g = np.mean(y2[mask])
            else:
                rho_g = 0
        else:
            rho_g = 0
    except Exception as e:
        rho_g = 0
        print("error fitting kmeans gas density")
        print(e, end="\n")

    ax1.set_xlabel("Z (Å)")
    ax1.set_ylabel(r"$\rho \ (\mathrm{g \ cm^{-3}})$")
    ax.legend(handlelength=1)
    # ax1.legend()
    fig.suptitle(f"{temp}K")
    ax.set_title(r"$\rho_{\ell} =$" + f" {np.round(rho_l/1000, 3)}")
    ax1.set_title(r"$\rho_{g} =$" + f" {np.round(rho_g/1000,3)}")

    fig.tight_layout()

    return fig, ax


def get_feature_distribution(
    x,
    phase,
    nbins=None,
    minmax=False,
    all_counts=False,
):
    x = x.copy().flatten()
    gas = phase.flatten() == 1
    liquid = ~gas

    N = gas.sum() + liquid.sum()
    if all_counts:
        ngas = N
        nliq = N
    else:
        ngas = gas.sum()
        nliq = liquid.sum()

    if minmax:
        x = utils.normalize_arr(x, norm="minmax")

    x_gas = x[gas]
    x_liq = x[liquid]
    if nbins is None:
        bins = 20
        bins_liq = 20
        bins_gas = 20
    elif isinstance(nbins, list):
        bins_liq, bins_gas = nbins
    else:
        bins = nbins
        bins_liq = nbins
        bins_gas = nbins

    _, bin_edges = np.histogram(x, bins=nbins)

    liq_counts, edges = np.histogram(x_liq, bins=bin_edges, density=False)
    liq_counts = liq_counts.astype(float) / nliq
    liq_counts[liq_counts == 0] = np.nan
    liq_center = (edges[1:] + edges[:-1]) / 2

    gas_counts, edges = np.histogram(x_gas, bins=bin_edges, density=False)
    gas_counts = gas_counts.astype(float) / ngas
    gas_counts[gas_counts == 0] = np.nan
    gas_center = (edges[1:] + edges[:-1]) / 2

    return liq_counts, liq_center, gas_counts, gas_center


def plot_feature_distribution(
    distribution,
    figsize=None,
    ax=None,
):
    from matplotlib.ticker import FormatStrFormatter

    liq_counts, liq_center, gas_counts, gas_center = distribution

    if ax is None:
        if figsize is None:
            figsize = (4, 2)
        fig, ax = plt.subplots(figsize=figsize, dpi=400)

    ax.plot(liq_counts, liq_center, color="navy")  # , linewidth=2)
    ax.plot(gas_counts, gas_center, color="darkred")  # , linewidth=2)

    ax.minorticks_on()
    ax.grid()
    # ax[j].set_ylabel("Probability", fontsize=14)
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    return ax


######################################### Archive ##########################################

# def average_distance(df, cluster_vars, start, norm=None):
#     if norm is not None:
#         centroids = df["centroids"]
#     else:
#         centroids = get_centroids(df)
#     nvars = len(cluster_vars)
#     liquid = df["molecule"]["liquid"][start:].flatten()
#     gas = df["molecule"]["gas"][start:].flatten()

#     avg_dist = np.zeros(3)
#     nt = df["nt"] - start
#     data = prep_data(
#         df,
#         cluster_vars,
#         start,
#         nt,
#         norm=norm,
#     )  # (nt*nmol, nvars)
#     gas_data = data[gas]
#     liq_data = data[liquid]
#     gas_disp = la.norm(gas_data - centroids[1], axis=1)
#     liq_disp = la.norm(liq_data - centroids[0], axis=1)
#     avg_dist[0] = np.mean(liq_disp)
#     avg_dist[1] = np.mean(gas_disp)
#     avg_dist[2] = np.mean(np.concatenate((liq_disp, gas_disp)))

#     return avg_dist


# def prep_data(
#     X,
#     cluster_vars,
#     minmax=True,
# ):
#     X = X.copy()
#     cluster_vars = np.array(cluster_vars)
#     nvars = len(cluster_vars)
#     cluster_data = np.empty_like(X)

#     for i, var in enumerate(cluster_vars):

#         feature = X[:, :, i]
#         if minmax:
#             feature = utils.normalize_arr(feature)
#         cluster_data[:, :, i] = feature

#     cluster_data = cluster_data.reshape(-1, nvars)

#     return cluster_data


# def classify_phase(
#     cluster_data,
#     cluster_vars,
#     k=2,
#     seed=0,
#     nstart=10,
#     tol=1e-12,
# ):

#     molecule_phase, centroids = kmeans.run_sk(cluster_data, k, seed, nstart, tol)
#     swap_centroids = kmeans.swap(cluster_vars, centroids)

#     if swap_centroids:
#         molecule_phase = 1 - molecule_phase
#         tmp = centroids[1].copy()
#         centroids[1] = centroids[0]
#         centroids[0] = tmp

#     to_return = {}
#     to_return["molecule phase"] = molecule_phase
#     to_return["centroids"] = centroids

#     return to_return
