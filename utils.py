from __future__ import annotations
import time
import numpy as np
import numba as nb
import psutil
import os
import sys
from pathlib import Path
import os
from typing import Dict, Any, List
from tqdm import tqdm
from read import weighted_average
import gc
import density


# @nb.njit
def normalize_arr(x, norm="minmax", quantiles=[75, 25]):
    if norm == "minmax":
        return (x - x.min()) / (x.max() - x.min())
    elif norm == "z":
        return (x - np.mean(x)) / np.std(x)
    elif norm == "iqr":
        q_high, q_low = np.percentile(x, quantiles)
        iqr = q_high - q_low
        if iqr == 0:
            return normalize_arr(x)
        return (x - np.mean(x)) / iqr

    else:
        return x


def generate_dz(lags, df):
    z = df["molecule"]["z"]

    for lag in lags:
        zi = np.zeros_like(z[lag:], dtype=bool)

        for mol in range(zi.shape[1]):
            zt = z[:, mol]

            # find where molecule crosses boundary
            zswitch = ((zt[lag:] < 0) & (zt[:-lag] > 0)) | (
                (zt[lag:] > 0) & (zt[:-lag] < 0)
            )
            zi[:, mol] = zswitch

        dz = np.zeros_like(zi, dtype=np.float32)
        dz[zi] = np.abs(z[lag:][zi] + z[:-lag][zi])
        dz[~zi] = np.abs(z[lag:][~zi] - z[:-lag][~zi])
        df["molecule"][f"dz_{lag}"] = dz
    return None


@nb.njit
def pbc(R, L):
    """
    Apply periodic boundary conditions to particle positions
    in a cubic box of side length L.

    Args:
        R (np.array): particle positions, shape (n, 3), (n, n, 3) or (nt, n, n, 3)
        L (np.array): side lengths of simulation box (3,) or one side length (float)
    Returns:
        np.array: particle positions, shape (n, n, 3)
    """

    # if single pos. vector
    if R.ndim == 1:
        R[0] -= L[0] * np.round(R[0] / L[0])
        R[1] -= L[1] * np.round(R[1] / L[1])
        R[2] -= L[2] * np.round(R[2] / L[2])

    # z coordinate
    # if R.ndim == 2:
    #     R[:] -= L[2] * np.round(R / L[2])

    # x y or z coordinates, L is a float
    if R.ndim == 2:
        R[:] -= L * np.round(R / L)

    # single timestep
    elif R.ndim == 3:
        R[:, :, 0] -= L[0] * np.round(R[:, :, 0] / L[0])
        R[:, :, 1] -= L[1] * np.round(R[:, :, 1] / L[1])
        R[:, :, 2] -= L[2] * np.round(R[:, :, 2] / L[2])

    # all timesteps
    elif R.ndim == 4:
        R[:, :, :, 0] -= L[0] * np.round(R[:, :, :, 0] / L[0])
        R[:, :, :, 1] -= L[1] * np.round(R[:, :, :, 1] / L[1])
        R[:, :, :, 2] -= L[2] * np.round(R[:, :, :, 2] / L[2])

    return R


def generate_switch_info(df, start=0, actime=False, center=True, offset=False):
    start = get_lag(df, start, actime)
    nt = df["nt"] - start
    molecule_phase = df["molecule"]["phase"][start:]
    switch_i = []
    for i in range(df["nmolecule"]):
        if not ((molecule_phase[:, i] == 0).all() or (molecule_phase[:, i] == 1).all()):
            switch_i.append(i)

    z = df["molecule"]["z"][start:].copy()
    if offset:
        lower_mask = df["molecule"]["lower_mask"][start:]
        z[lower_mask] -= df["offset"]

    if center:
        com = np.mean(df["atom"]["z"][start:], axis=1)
        z -= com[:, np.newaxis]
        L = get_L(df)
        z = pbc(z, L[-1])

    # Z traces of any molecule that has switched phases at any timestep (nt, nswitch)
    switch_z = z[:, switch_i]
    switch_phase = molecule_phase[:, switch_i]  # The corresponding phases
    # switch_z[switch_z < 0] -= df["offset"]

    gas_liquid_z = []  # z pos at which a gas changes to liquid
    gas_liquid_i = []  # molecular indices when molecules switch from gas to liquid
    liquid_gas_z = []  # z pos at which a liquid changes to a gas
    liquid_gas_i = []  # molecular indices when molecules switch from liquid to gas
    gas_liquid_t = []  # times at which a gas to liquid transition is made
    liquid_gas_t = []  # times at which a liquid to gas transition is made

    for i in range(switch_z.shape[1]):

        last_phase = switch_phase[0, i]

        for t in range(1, nt):

            phase = switch_phase[t, i]

            if phase != last_phase:

                if last_phase == 0:
                    liquid_gas_z.append(switch_z[t, i])
                    liquid_gas_t.append(t + start)
                    liquid_gas_i.append(i)

                elif last_phase == 1:
                    gas_liquid_z.append(switch_z[t, i])
                    gas_liquid_t.append(t + start)
                    gas_liquid_i.append(i)

            last_phase = phase

    gas_liquid_z = np.array(gas_liquid_z)
    liquid_gas_z = np.array(liquid_gas_z)
    gas_liquid_t = np.array(gas_liquid_t)
    liquid_gas_t = np.array(liquid_gas_t)
    gas_liquid_i = np.array(gas_liquid_i)
    liquid_gas_i = np.array(liquid_gas_i)
    # any timestep at which A switch happens
    switch_t = np.sort(np.unique(np.concatenate([liquid_gas_t, gas_liquid_t])))

    if df["metal_type"] != -1:
        switch_z = np.abs(switch_z)
        gas_liquid_z = np.abs(gas_liquid_z)
        liquid_gas_z = np.abs(liquid_gas_z)

    df["molecule"]["switch_i"] = switch_i
    df["molecule"]["switch_z"] = switch_z
    df["molecule"]["gas_liquid_z"] = gas_liquid_z
    df["molecule"]["liquid_gas_z"] = liquid_gas_z
    df["molecule"]["gas_liquid_t"] = gas_liquid_t
    df["molecule"]["liquid_gas_t"] = liquid_gas_t
    df["molecule"]["gas_liquid_i"] = gas_liquid_i
    df["molecule"]["liquid_gas_i"] = liquid_gas_i
    df["molecule"]["switch_t"] = switch_t

    n_to_gas = np.zeros(nt + start, dtype=int)
    n_to_gas[: start + 1] = 0
    n_to_liquid = np.zeros(nt + start, dtype=int)
    n_to_liquid[: start + 1] = 0
    nliquid = df["molecule"]["nliquid"]  # [start:]
    ngas = df["molecule"]["ngas"]  # [start:]
    rate_to_gas = np.zeros(nt + start)
    rate_to_liquid = np.zeros(nt + start)
    prob_to_gas = np.zeros(nt + start)
    prob_to_liquid = np.zeros(nt + start)

    for t in range(start + 1, nt):

        n_to_gas[t] = int((liquid_gas_t == t).sum())
        n_to_liquid[t] = int((gas_liquid_t == t).sum())

        if nliquid[t - 1] > 0:
            #  per femtosecond * 1e6 -> per nanosecond
            rate_to_gas[t] = n_to_gas[t] / nliquid[t - 1] / df["dt"] * 1e6
            prob_to_gas[t] = n_to_gas[t] / nliquid[t - 1]
        else:
            rate_to_gas[t] = 0

        if ngas[t - 1] > 0:
            rate_to_liquid[t] = n_to_liquid[t] / ngas[t - 1] / df["dt"] * 1e6
            prob_to_liquid[t] = n_to_liquid[t] / ngas[t - 1]
        else:
            rate_to_liquid[t] = 0

    df["molecule"]["n_to_gas"] = n_to_gas
    df["molecule"]["n_to_liquid"] = n_to_liquid
    df["molecule"]["rate_to_gas"] = rate_to_gas
    df["molecule"]["rate_to_liquid"] = rate_to_liquid
    df["molecule"]["prob_to_gas"] = prob_to_gas
    df["molecule"]["prob_to_liquid"] = prob_to_liquid

    return None


def print_memory(path=None):

    if path:
        file_size = os.path.getsize(path)
        print(f"file size: {file_size / 1024**3:.2f} GB")

    total_memory = psutil.virtual_memory().total / 1024**3
    print(f"Total Memory: {total_memory:.2f} GB")

    available_memory = psutil.virtual_memory().available
    print(f"Available Memory: {available_memory / 1024**3:.2f} GB")

    process = psutil.Process(os.getpid())
    used_memory = process.memory_info().rss / (1024**3)
    print(f"Used Memory: {used_memory:.2f} GB")

    return None


def getsize(df):
    s1 = 0

    for key in list(df["atom"].keys()):
        data = df["atom"][key]
        size = sys.getsizeof(data) / 1e9
        s1 += size

    s2 = 0

    for key in list(df["molecule"].keys()):
        data = df["molecule"][key]
        size = sys.getsizeof(data) / 1e9
        s2 += size

    s1 = np.round(s1, 3)
    s2 = np.round(s2, 3)
    s3 = np.round(s1 + s2, 3)

    print(f"df_atom: {s1} GB df_molecule: {s2} GB Total: {s3} GB")

    return None


def parse(df, expr, mode="molecule"):

    df_mode = df[mode]

    op_map = {
        "log": np.log10,
        "norm": normalize_arr,
        "abs": np.abs,
        "sqrt": np.sqrt,
    }
    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    data = df_mode[var].copy()
    if mode == "atom":
        data = data[:, df["non_metal"]]

    for op in ops:
        if op not in list(op_map.keys()):
            print("Supported operations: ", list(op_map.keys()))
            raise ValueError(f"Invalid operation: {op}")
        if op == "log":
            data = data.astype(np.float32) + 1e-6
        f = op_map[op]
        data = f(data)

    if data.shape[0] < df["nt"]:
        lag = df["nt"] - data.shape[0]
        if data.ndim == 2:
            data = np.pad(data, ((lag, 0), (0, 0)), "constant", constant_values=0)
        elif data.ndim == 1:
            data = np.pad(data, (lag, 0), "constant", constant_values=0)

    return data


def strip_var(var):
    var_stripped = var.strip().split("(")
    var_stripped = [item.strip(")").strip() for item in var_stripped if item][-1]

    return var_stripped


def parse_label(expr):
    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    if len(var.split("_")) > 1:
        var = var.split("_")
        label = "\\text{" + var[0] + "}" + "_{" + var[1] + "}"
    else:
        label = "\\text{" + var + "}"
    for op in ops:
        if op == "abs":
            label = f"\\lvert {label} \\rvert"

        elif op == "norm":
            label += "_{\\text{norm}}"

        elif op == "log":
            label = f"\\log ({label})"

        elif op == "sqrt":
            label = f"\\sqrt ({label})"

    label = "$" + label + "$"

    return label


def parse_ylabel(expr):

    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    info = ""
    if len(var.split("_")) > 1:
        var = var.split("_")
        feature = var[0]
        info = int(var[-1])
    else:
        feature = var

    feature_units = {
        "coordination": "n_{molecule}",
        "pe": r"U \ \text{(kcal/mol)}",
        "displacement": rf"\Delta s_{{{info * 50}}} \ (\text{{Å}})",
        "ke": r"\text{Kinetic Energy (kcal/mol)}",
        "lt": "K",
        "density": rf"\rho_{{{info}}} \ \mathrm{{(g \ cm^{{-3}})}}",
    }
    units = feature_units[feature]
    label = units

    for op in ops:
        if op == "abs":
            label = f"\\lvert {units} \\rvert"

        elif op == "norm":
            label = units + "_{\\text{norm}}"

        elif op == "log":
            label = f"\\log ({units})"

        elif op == "sqrt":
            label = f"\\sqrt ({units})"

    label = "$" + label + "$"

    return label


def concat_labels(labels):
    concat = r""
    new_labels = []
    for label in labels:
        new_labels.append(parse_label(label))

    for i, label in enumerate(new_labels):
        if i == len(labels) - 1:
            concat += label.strip("$")
        else:
            concat += label.strip("$") + r", \ "

    concat = r"[" + concat + r"]"
    return concat


def format(number):
    if number == 0 or number == -1:
        return ""
    magnitude = np.floor(np.log10(number))

    if 3 <= magnitude <= 5:
        number /= 10**3
        if magnitude == 5 or magnitude == 4:
            return f"{number:.0f}K"
        else:
            return f"{number:.1f}K"
    elif magnitude == 2:
        return f"{number:.0f}"
    elif magnitude == 1:
        return f"{number:.1f}"
    elif magnitude == 0:
        return f"{number:.2f}"
    elif magnitude == -1:
        return f"{number:.2f}"
    elif magnitude == -2:
        return f"{number:.2f}"
    elif magnitude <= -3:
        number *= 10**-magnitude
        return f"{number:.0f}e{magnitude:.0f}"


def get_pos(df, mode="molecule"):
    x = df[mode]["x"]
    y = df[mode]["y"]
    z = df[mode]["z"]
    r = np.stack((x, y, z), axis=-1)

    return r


def get_L(df):
    L = df["bounds"][:, 1] * 2
    return L


def get_v(df):
    vx = df["molecule"]["vx"]
    vy = df["molecule"]["vy"]
    vz = df["molecule"]["vz"]
    v = np.stack((vx, vy, vz), axis=-1)
    return v


def get_lag(df, start=0, actime=False, features=None):

    if "cluster_vars" not in df.keys() and features is None:
        return 0

    if features is None:
        features = df["cluster_vars"]
    lag = 0
    for var in features:
        if "displacement" in var or "dz" in var:
            split = var.split("_")[-1]
            try:
                curr_lag = int(split[:2])
            except:
                curr_lag = int(split[:1])

            if curr_lag > lag:
                lag = curr_lag
    if start > lag:
        lag = start

    if actime:
        diff = df["actime"] - lag
        if diff > 0:
            lag += diff

    return lag


def phase_frac(
    df, phase, mode="molecule", std=True, start=0, actime=True, time_avg=True
):
    start = get_lag(df, start, actime)
    phase_map = {0: "nliquid", 1: "ngas"}
    n_phase = phase_map[phase]
    n = df["molecule"][n_phase][start:]
    if mode == "atom":
        n *= df["natom_per_molecule"]
        frac = n / (df["natom"] - df["nmetal"])
    else:
        frac = n / df["nmolecule"]
    if std:
        err = np.std(frac)
    if time_avg:
        frac = np.mean(frac)
    if std:
        return frac, err
    else:
        return frac


def get_t_ns(df, t):
    timesteps = df["timesteps"]
    t0 = timesteps[0]
    dt = df["dt"]
    t_ns = (t * dt + t0) / 1e6

    return t_ns


def get_info(path):
    info = path.split("/")[-1]
    molecule_name = info.split("_")[1]
    temp = info.find("K")
    temp = int(info[temp - 3 : temp])
    return molecule_name, temp


def molecule_size(df):

    z = df["atom"]["z"][0][df["reference_molecule_mask"]]
    x = df["atom"]["x"][0][df["reference_molecule_mask"]]
    y = df["atom"]["y"][0][df["reference_molecule_mask"]]

    m = 0
    n = z.shape[0]
    for i in range(n):
        p1 = np.array([x[i], y[i], z[i]])
        for j in range(i, n):
            p2 = np.array([x[j], y[j], z[j]])
            d = np.linalg.norm(p1 - p2)
            if d > m:
                m = d
    return m


def rad_gyration(df):

    ref_mask = df["reference_molecule_mask"]

    z = df["atom"]["z"][0][ref_mask]
    x = df["atom"]["x"][0][ref_mask]
    y = df["atom"]["y"][0][ref_mask]

    mol_idx = df["atom"]["molecule_id"][df["reference_molecule_mask"]][0]
    mol_mask = df["molecule"]["id"] == mol_idx
    x_com = df["molecule"]["x"][0][mol_mask]
    y_com = df["molecule"]["y"][0][mol_mask]
    z_com = df["molecule"]["z"][0][mol_mask]
    com = np.array([x_com, y_com, z_com]).flatten()
    atom_masses = df["atom"]["mass"][ref_mask]

    s = 0
    n = z.shape[0]
    for i in range(n):
        mi = atom_masses[i]
        pi = np.array([x[i], y[i], z[i]])
        di = np.linalg.norm(pi - com) ** 2
        s += mi * di

    return np.sqrt(s / np.sum(atom_masses))


def write_dump(
    df: dict,
    filename: str | Path,
    *,
    precision: int = 6,
    mode: str = "molecule",
    disable=True,
) -> None:
    """
    Parameters
    ----------
    df        : molecule dictionary
    filename  : path to the output *.dump file.

    precision : digits after the decimal point for each floating value.
    """
    # Basic validation --------------------------------------------------------
    valid_keys = [
        "id",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "lt",
        "ke",
        "pe",
        "q",
        "speed",
        "mass",
        "phase",
        "displacement",
        "coordination",
        "density",
        "molecule_id",
        "type",
    ]
    int_fmt = {
        "molecule_id": "%d",
        "id": "%d",
        "phase": "%d",
        "type": "%d",
    }
    float_fmt = f"%.{precision}f"  # default for other floats
    fmt_line = []

    df_mode = df[mode]
    keys = []
    nt = df["nt"]
    L = df["bounds"]
    nmol = df["natom"] if mode == "atom" else df["nmolecule"]
    keys = [key for key in df_mode if key.split("_")[0] in valid_keys]
    data = np.zeros((nt, nmol, len(keys)), dtype=np.float32)
    for i, key in enumerate(keys):
        fmt = int_fmt.get(key, (float_fmt))
        fmt_line.append(fmt)
        if key == "id":
            x = np.repeat(df_mode[key], nt).reshape(nmol, nt).T + 1
        elif key == "mass" and mode == "molecule":
            x = np.ones((nt, nmol)) * df_mode[key]
        elif key == "mass" and mode == "atom":
            x = np.repeat(df_mode[key], nt).reshape(nmol, nt).T + 1
        elif key == "molecule_id" or key == "type":
            x = np.repeat(df_mode[key], nt).reshape(nmol, nt).T + 1
        else:
            x = parse(df, key, mode=mode)
        data[:, :, i] = x

    header = " ".join(keys)
    path = Path(filename).with_suffix(".dump")
    with path.open("w") as fh:
        for t in tqdm(range(nt), disable=disable):
            # ---- LAMMPS dump headers ----------------------------------------
            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{t}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{nmol}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            fh.write(f"{L[0, 0]} {L[0, 1]}\n")
            fh.write(f"{L[1, 0]} {L[1, 1]}\n")
            fh.write(f"{L[2, 0]} {L[2, 1]}\n")
            fh.write(f"ITEM: ATOMS {header}\n")

            block = data[t]
            np.savetxt(
                fh,
                block,
                fmt=fmt_line,
                delimiter=" ",
            )
    if not disable:
        print(f"Wrote {nt} frames, {nmol} molecules per frame → '{path}'.")
    return None


def write_phase(
    dump_path: str | Path,
    df: Dict[str, Any],
    *,
    batch_size: int = 50,  # timesteps to buffer before write
    tmp_suffix: str = ".phase.tmp",  # temp file before atomic replace
) -> Path:
    """
    Add *or replace* the integer `phase` column in an ASCII LAMMPS dump
    (mode = 'atom'). The original file is atomically replaced.

    Parameters
    ----------
    dump_path : path to the existing dump file.
    df        : data dictionary.  Needs
                  df['atom']['id']      – (natom,) *sorted* ids
                  parse(df,'phase')     – (nt, natom) phases ordered by id
    batch_size: number of timesteps kept in RAM before flushing to disk.

    Returns
    -------
    Path to the updated dump file (same name, new contents).
    """
    dump_path = Path(dump_path).expanduser()
    tmp_path = dump_path.with_suffix(dump_path.suffix + tmp_suffix)

    # ----- pull arrays from df ------------------------------------------- #
    ids_sorted = np.asarray(df["atom"]["id"], dtype=np.int32)  # (natom,)
    phase_all = parse(df, "phase", mode="atom").astype(np.int8)  # (nt, natom)
    nt, natom = phase_all.shape

    # quick id → row_index map (works for positive, dense-ish IDs)
    max_id = ids_sorted.max()
    id2row = np.empty(max_id + 1, dtype=np.int32)
    id2row[ids_sorted] = np.arange(natom, dtype=np.int32)

    # ----- streamed read → write ---------------------------------------- #
    with dump_path.open("r", buffering=4_194_304) as src, tmp_path.open(
        "w", buffering=4_194_304
    ) as dst:

        out_buf: List[str] = []
        ts_idx = 0  # current timestep index
        buffered_ts = 0  # how many ts held in out_buf buffer

        while True:
            line = src.readline()
            if not line:  # EOF
                break

            if line.startswith("ITEM: ATOMS"):
                tokens = line.strip().split()
                atom_fields = tokens[2:]  # after "ITEM: ATOMS"
                id_col = atom_fields.index("id")  # required
                # locate phase column if present
                phase_present = "phase" in atom_fields
                if phase_present:
                    phase_col = atom_fields.index("phase")
                    out_buf.append(line)  # header unchanged
                else:
                    phase_col = None
                    out_buf.append(line.rstrip("\n") + " phase\n")

                # ------------ read natom atom lines ---------------------- #
                atom_lines = [src.readline() for _ in range(natom)]

                # vectorised ID extraction
                ids_block = (
                    np.fromiter(
                        (int(al.split()[id_col]) for al in atom_lines),
                        dtype=np.int32,
                        count=natom,
                    )
                    - 1
                )

                # corresponding phase values
                ph_vals = phase_all[ts_idx, id2row[ids_block]]
                ph_str = np.char.mod("%d", ph_vals)  # array of strings

                # ------------ patch / append per line ------------------- #
                if phase_present:
                    # replace existing value
                    for i, l in enumerate(atom_lines):
                        parts = l.rstrip().split()
                        parts[phase_col] = ph_str[i]
                        out_buf.append(" ".join(parts) + "\n")
                else:
                    # just append
                    out_buf.extend(
                        f"{atom_lines[i].rstrip()} {ph_str[i]}\n" for i in range(natom)
                    )

                ts_idx += 1
                buffered_ts += 1

                if buffered_ts >= batch_size:
                    dst.writelines(out_buf)
                    out_buf.clear()
                    buffered_ts = 0
            else:
                out_buf.append(line)

        # flush trailing timesteps
        if out_buf:
            dst.writelines(out_buf)

    # ----- atomic replace ------------------------------------------------- #
    os.replace(tmp_path, dump_path)
    return None


def write_switch_info(df, path, centroids=None):
    switch_t = df["molecule"]["switch_t"]
    gas_liquid_i = df["molecule"]["gas_liquid_i"]
    liquid_gas_i = df["molecule"]["liquid_gas_i"]
    gas_liquid_t = df["molecule"]["gas_liquid_t"]
    liquid_gas_t = df["molecule"]["liquid_gas_t"]
    n_to_gas = parse(df, "n_to_gas")
    n_to_liquid = parse(df, "n_to_liquid")
    nliquid = parse(df, "nliquid")
    ngas = parse(df, "ngas")
    if centroids is None:
        centroids = df["centroids"]
    cluster_vars = df["cluster_vars"]
    x = df["molecule"]["x"]
    y = df["molecule"]["y"]
    z = df["molecule"]["z"]
    timesteps = df["timesteps"]
    rate_to_gas = parse(df, "rate_to_gas")
    rate_to_liquid = parse(df, "rate_to_liquid")
    prob_to_gas = parse(df, "prob_to_gas")
    prob_to_liquid = parse(df, "prob_to_liquid")

    with open(path, "w+") as f:
        header = f"HEADER {cluster_vars} mu_liq({list(centroids[0])}) mu_gas({list(centroids[-1])})"
        f.write(header + "\n")

        for t in range(df["nt"]):
            t_header = f"TIMESTEP {timesteps[t]} nliq {nliquid[t]} ngas {ngas[t]} n_to_gas {n_to_gas[t]} n_to_liq {n_to_liquid[t]}"
            t_header += f" rate_to_gas {np.round(rate_to_gas[t], 4)} rate_to_liq {np.round(rate_to_liquid[t], 4)}"
            t_header += f" prob_to_gas {np.round(prob_to_gas[t], 4)} prob_to_liq {np.round(prob_to_liquid[t], 4)}"
            # t_header += " #per nanosecond"
            f.write(t_header + "\n")
            if t not in switch_t:
                continue

            info_header = "ID X Y Z liq gas"
            f.write(info_header + "\n")

            curr_gas_liquid_i = gas_liquid_i[gas_liquid_t == t]
            curr_liquid_gas_i = liquid_gas_i[liquid_gas_t == t]

            sort = np.sort(np.concatenate([curr_gas_liquid_i, curr_liquid_gas_i]))

            for id in sort:
                X = str(np.round(x[t][id], 4))
                Y = str(np.round(y[t][id], 4))
                Z = str(np.round(z[t][id], 4))
                line = f"{id} {X} {Y} {Z}"
                if id in curr_gas_liquid_i:
                    line += " 1 0\n"
                else:
                    line += " 0 1\n"
                f.write(line)

    return None


def read_switch_info(path):

    timestep_info_cols = [
        "timestep",
        "nliq",
        "ngas",
        "n_to_gas",
        "n_to_liquid",
        "rate_to_gas",
        "rate_to_liquid",
        "prob_to_gas",
        "prob_to_liquid",
    ]
    mol_switch_cols = ["id", "x", "y", "z", "liquid", "gas"]
    timestep_info = {}
    mol_switch = {}
    centroids = {}
    with open(path, "r") as f:

        lines = f.readlines()
        n = len(lines) - 1
        timestep_info_arr = []
        mol_switch_arr = []
        for i, line in enumerate(lines):

            if line.startswith("HEADER"):

                s = line.find("[")
                e = line.find("]", s)
                cluster_vars = eval(line[s : e + 1])
                s = line.find("[", e)
                e = line.find("]", s)
                mu_liq = eval(line[s : e + 1])
                s = line.find("[", e)
                e = line.find("]", s)
                mu_gas = eval(line[s : e + 1])

                centroids["liquid"] = mu_liq
                centroids["gas"] = mu_gas
                centroids["vars"] = cluster_vars

            elif line.startswith("TIMESTEP"):
                timestep_info_arr.append(np.array(line.split()[1::2], float))

            else:

                try:
                    mol_switch_ = np.array(line.split(), float)
                    mol_switch_arr.append(mol_switch_)

                except:
                    continue

        timestep_info_arr = np.array(timestep_info_arr)
        mol_switch_arr = np.array(mol_switch_arr)

        for j in range(timestep_info_arr.shape[1]):

            timestep_info[timestep_info_cols[j]] = timestep_info_arr[:, j]

        for j in range(mol_switch_arr.shape[1]):
            mol_switch[mol_switch_cols[j]] = mol_switch_arr[:, j]

    return timestep_info, mol_switch, centroids


@nb.njit
def unwrap_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct unwrapped Cartesian coordinates.

    Parameters
    ----------
    x, y, z : (nt, natom) float64 arrays
        Wrapped positions in Å (or whatever unit the dump writes).
    ix, iy, iz : (nt, natom) int32/64 arrays
        Cumulative image flags returned by LAMMPS (`image` or `ix`, `iy`, `iz`).
    bounds : (3, 2) float64 array
        [[xlo, xhi], [ylo, yhi], [zlo, zhi]] of the simulation box for *all* frames.
        (If the box is time-dependent, pass an array of shape (nt, 3, 2) instead.)

    Returns
    -------
    pos : (nt, natom, 3) float64 array
        Unwrapped coordinates.
    """
    # Box lengths ΔLx, ΔLy, ΔLz
    box_len = bounds[:, 1] - bounds[:, 0]  # shape (3,)
    # Broadcast the image counters onto the wrapped coordinates
    pos_x = x + ix * box_len[0]
    pos_y = y + iy * box_len[1]
    pos_z = z + iz * box_len[2]
    return np.stack((pos_x, pos_y, pos_z), axis=2)  # (nt, natom, 3)


def generate_displacement(
    df: Dict,
    lags: np.ndarray,
    keep_atom: bool = False,
    to_print=False,
) -> np.ndarray:

    if to_print:
        s = time.time()
        print("\nCalculating dipslacement...", end="")
    bounds = df["bounds"]
    nt = df["nt"]
    nmolecule = df["nmolecule"]
    natom_per_molecule = df["natom_per_molecule"]
    x, y, z = df["atom"]["x"], df["atom"]["y"], df["atom"]["z"]
    ix, iy, iz = df["atom"]["ix"], df["atom"]["iy"], df["atom"]["iz"]
    pos = unwrap_coordinates(x, y, z, ix, iy, iz, bounds)  # (nt, natom, 3)

    # mol = df["atom"]["molecule_id"] == 0
    # xx = x[0, mol]
    # yy = y[0, mol]
    # zz = z[0, mol]

    # coords = np.stack((xx, yy, zz), axis=1)  # (natom_per_molecule, 3)

    # xx1 = x[1, mol]
    # yy1 = y[1, mol]
    # zz1 = z[1, mol]
    # coords1 = np.stack((xx1, yy1, zz1), axis=1)  # (natom_per_molecule, 3)

    # m = df["atom"]["mass"][:natom_per_molecule]

    # disp = (m[:, None] * (coords1 - coords)).sum(axis=0) / np.sum(m)
    # print(disp.shape)
    # print(np.linalg.norm(disp))

    # disp_vec = coords1 - coords
    # disp1 = np.linalg.norm(disp_vec, axis=1)  # (natom_per_molecule,)

    # weighted_disp = np.sum(m * disp1) / np.sum(m)
    # print(weighted_disp)
    # dx = (xx1 - xx) * m
    # dy = (yy1 - yy) * m
    # dz = (zz1 - zz) * m

    # disp = np.sqrt((dx**2 + dy**2 + dz**2) * m) / np.sum(m)
    # print(disp)

    non_metal = df["non_metal"]

    for lag in lags:
        disp_vec = pos[lag:] - pos[:-lag]  # (nt-lag, natom, 3)
        # disp = np.linalg.norm(disp_vec, axis=2)  # (nt-lag, natom)
        # prepend zeros
        # disp = np.pad(disp, ((lag, 0), (0, 0)), mode="constant", constant_values=0)
        # mol_disp = read.molecule_average(disp, nt, nmolecule, natom_per_molecule)

        # mol_disp = read.molecule_average(disp, nt - lag, nmolecule, natom_per_molecule)
        mol_disp_x = weighted_average(
            disp_vec[:, non_metal, 0],
            df["atom"]["mass"][non_metal],
            nt - lag,
            nmolecule,
            natom_per_molecule,
            df["molecule"]["mass"],
        )

        mol_disp_y = weighted_average(
            disp_vec[:, non_metal, 1],
            df["atom"]["mass"][non_metal],
            nt - lag,
            nmolecule,
            natom_per_molecule,
            df["molecule"]["mass"],
        )

        mol_disp_z = weighted_average(
            disp_vec[:, non_metal, 2],
            df["atom"]["mass"][non_metal],
            nt - lag,
            nmolecule,
            natom_per_molecule,
            df["molecule"]["mass"],
        )

        mol_disp = np.stack((mol_disp_x, mol_disp_y, mol_disp_z), axis=1)

        mol_disp = np.linalg.norm(mol_disp, axis=1)

        # mol_disp = weighted_average(
        #     disp[:, non_metal],
        #     df["atom"]["mass"][non_metal],
        #     nt - lag,
        #     nmolecule,
        #     natom_per_molecule,
        #     df["molecule"]["mass"],
        # )

        mol_disp = np.pad(
            mol_disp, ((lag, 0), (0, 0)), mode="constant", constant_values=0
        )

        df["molecule"][f"displacement_{lag}"] = mol_disp
        del mol_disp_x, mol_disp_y, mol_disp_z
        if keep_atom:
            df["atom"][f"displacement_{lag}"] = np.linalg.norm(disp_vec, axis=2)
            del disp_vec
            gc.collect()
        else:
            del disp_vec

            gc.collect()

    gc.collect()

    if to_print:
        e = time.time()
        dt = e - s
        print(f" Done in {dt:.2f}s")

    return None


def feats_z(df, cluster_vars, dz=2, norm="minmax"):

    z = df["molecule"]["z"][1:].copy()
    com = np.mean(df["atom"]["z"][1:], axis=1)
    z -= com[:, np.newaxis]
    L = get_L(df)
    z = pbc(z, L[-1])

    # dz = 4
    zmax_pos, zmax_neg = density.get_min_max(z)
    zmin = np.max(np.min(np.abs(z), axis=1))
    zmax = zmax_pos if zmax_pos <= zmax_neg else zmax_neg
    nbins = int(np.round((zmax - zmin) / dz, 0))
    hist_range = (zmin, zmax)
    bins = np.linspace(hist_range[0], hist_range[1], nbins + 1, endpoint=True)
    zbin = (bins[1:] + bins[:-1]) / 2
    z = np.abs(z)

    ind = np.digitize(z, bins)

    yy = np.zeros((len(cluster_vars), nbins))
    for j, var in enumerate(cluster_vars):
        y = parse(df, var)[1:]
        if isinstance(norm, str):
            y = normalize_arr(y, norm)
        elif isinstance(norm, np.ndarray):
            ymin = norm[j, 0]
            ymax = norm[j, 1]
            y = (y - ymin) / (ymax - ymin)

        s = np.zeros(nbins)
        n = np.zeros(nbins)
        for i in range(1, len(bins)):

            mask = np.where(ind == i)
            s[i - 1] += np.sum(y[mask])
            n[i - 1] += (ind == i).sum()
        yy[j] = s / n

    return zbin, yy


def print_rows(arrays: dict[str, np.ndarray], round: int = 2):
    # assume all arrays same length
    n = len(next(iter(arrays.values())))
    # pre‐format every entry to string
    str_vals = {
        name: [str(np.round(x, round)) if not isinstance(x, str) else x for x in arr]
        for name, arr in arrays.items()
    }
    if "T" in str_vals.keys():
        str_vals["T"] = [str(np.round(float(temp), round)) for temp in str_vals["T"]]
    # compute column widths
    col_widths = [max(len(str_vals[name][i]) for name in arrays) + 1 for i in range(n)]
    # print each row
    for name, vals in str_vals.items():
        row = name.rjust(max(len(n) for n in arrays)) + ": [ "
        row += " ".join(v.rjust(w) for v, w in zip(vals, col_widths))
        row += " ]"
        print("\t", row)


######################################### Archive ##########################################

# def dump_phase(df, infile, outfile):
#     from ovito.io import import_file
#     from ovito.io import export_file

#     atom_phase = df["atom"]["phase"]

#     def modify(t, data):
#         data.particles_.create_property("phase", np.int8)
#         ids = data.particles["Particle Identifier"][...]
#         sort = np.argsort(ids)
#         inv_sort = np.argsort(sort)
#         data.particles["phase"][...] = atom_phase[t].flatten()[inv_sort]

#     pipeline = import_file(infile)
#     pipeline.modifiers.append(modify)

#     columns = list(pipeline.compute().particles.keys())
#     export_file(pipeline, outfile, "lammps/dump", multiple_frames=True, columns=columns)

#     return None


# from ovito.io import import_file
# from ovito.modifiers import CalculateDisplacementsModifier
# from ovito.modifiers import UnwrapTrajectoriesModifier

# pipeline = import_file(data_path)
# data = pipeline.compute()
# list(data.particles.keys())
# image_modifier = UnwrapTrajectoriesModifier()
# disp_modifer = CalculateDisplacementsModifier(
#     frame_offset=-1, minimum_image_convention=False, use_frame_offset=True
# )
# pipeline.modifiers.append(image_modifier)
# pipeline.modifiers.append(disp_modifer)

# X = np.zeros((10, 100056, 3))
# ids = np.zeros((10, 100056))
# for frame in range(pipeline.num_frames):
#     if frame == 0:
#         continue
#     if frame == 10:
#         break
#     data = pipeline.compute(frame)
#     displacements = data.particles["Displacement"]
#     id = data.particles["Particle Identifier"][...]
#     order = np.argsort(id)
#     X[frame] = displacements[order]
#     # ids[frame] = id


# def generate_displacement(lags, df):
# x = df["molecule"]["x"]
# y = df["molecule"]["y"]
# z = df["molecule"]["z"]

# for lag in lags:
#     xi = np.zeros_like(x[lag:], dtype=bool)
#     yi = np.zeros_like(y[lag:], dtype=bool)
#     zi = np.zeros_like(z[lag:], dtype=bool)

#     for mol in range(xi.shape[1]):
#         xt = x[:, mol]
#         yt = y[:, mol]
#         zt = z[:, mol]

#         xswitch = ((xt[lag:] < 0) & (xt[:-lag] > 0)) | (
#             (xt[lag:] > 0) & (xt[:-lag] < 0)
#         )
#         yswitch = ((yt[lag:] < 0) & (yt[:-lag] > 0)) | (
#             (yt[lag:] > 0) & (yt[:-lag] < 0)
#         )
#         zswitch = ((zt[lag:] < 0) & (zt[:-lag] > 0)) | (
#             (zt[lag:] > 0) & (zt[:-lag] < 0)
#         )

#         xi[:, mol] = xswitch
#         yi[:, mol] = yswitch
#         zi[:, mol] = zswitch

#     dx = np.zeros_like(xi, dtype=np.float32)
#     dy = np.zeros_like(yi, dtype=np.float32)
#     dz = np.zeros_like(zi, dtype=np.float32)

#     dx[xi] = np.abs(x[lag:][xi] + x[:-lag][xi])
#     dy[yi] = np.abs(y[lag:][yi] + y[:-lag][yi])
#     dz[zi] = np.abs(z[lag:][zi] + z[:-lag][zi])

#     dx[~xi] = np.abs(x[lag:][~xi] - x[:-lag][~xi])
#     dy[~yi] = np.abs(y[lag:][~yi] - y[:-lag][~yi])
#     dz[~zi] = np.abs(z[lag:][~zi] - z[:-lag][~zi])

#     net_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
#     df["molecule"][f"displacement_{lag}"] = net_displacement

# return None
