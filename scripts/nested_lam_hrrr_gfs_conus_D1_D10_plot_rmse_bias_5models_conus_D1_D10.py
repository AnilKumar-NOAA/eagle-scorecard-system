#!/usr/bin/env python3

from pathlib import Path
import os
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


SYSTEM_BASE = Path("/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard_system")
DATA_BASE = Path(os.environ.get("SCORECARD_SYSTEM_DATA_DIR", SYSTEM_BASE / "data"))
OUTPUT_DIR = Path(os.environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", SYSTEM_BASE / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FHRS = list(range(24, 241, 6))
XTICK_FHRS = list(range(24, 241, 24))
XTICK_LABELS = [f"D{int(f/24)}" for f in XTICK_FHRS]

LEVELS = [250, 500, 850]

MODEL_COLORS = {
    "Nested-EAGLE-LAM": "#1f77b4",  # blue
    "HRRR": "#ff7f0e",              # orange
    "GFS": "#000000",               # black
    "AIGFS": "#2ca02c",             # green
    "ECMWF IFS": "#d62728",         # red
}

MODEL_LABELS = {
    "Nested-EAGLE-LAM": "Nested-EAGLE LAM (~6 km)",
    "HRRR": "HRRR (~6 km)",
    "GFS": "GFS CONUS (0.25 deg)",
    "AIGFS": "AIGFS CONUS (0.25 deg)",
    "ECMWF IFS": "ECMWF-IFS CONUS (0.25 deg)",
}

MODELS = {
    "Nested-EAGLE-LAM": {
        "dir": DATA_BASE / "nested_eagle_lam_2025",
        "patterns": [
            "{metric}.convobs.nested-lam.2025-*_to_2025-*.nc",
            "{metric}.convobs.nested_lam.2025-*_to_2025-*.nc",
        ],
    },
    "HRRR": {
        "dir": DATA_BASE / "hrrr_2025",
        "patterns": [
            "{metric}.convobs.lam*.2025-*_to_2025-*.nc",
            "{metric}.convobs.hrrr*.2025-*_to_2025-*.nc",
        ],
    },
    "GFS": {
        "dir": DATA_BASE / "gfs_zarr_2025",
        "patterns": [
            "{metric}.convobs.global.conus.2025-*_to_2025-*.nc",
            "{metric}.convobs.global.2025-*_to_2025-*.nc",
        ],
    },
    "AIGFS": {
        "dir": DATA_BASE / "aigfs_2025",
        "patterns": [
            "{metric}.convobs.global.conus.2025-*_to_2025-*.nc",
            "{metric}.convobs.global.2025-*_to_2025-*.nc",
        ],
    },
    "ECMWF IFS": {
        "dir": DATA_BASE / "ecmwf_ifs_2025",
        "patterns": [
            "{metric}.convobs.global.conus.2025-*_to_2025-*.nc",
            "{metric}.convobs.global.2025-*_to_2025-*.nc",
        ],
    },
}

VAR_ALIASES = {
    "2m_temperature": ["2m_temperature"],
    "2m_specific_humidity": ["2m_specific_humidity"],
    "surface_pressure": ["surface_pressure"],
    "10m_u_wind": ["10m_zonal_wind"],
    "10m_v_wind": ["10m_meridional_wind"],
    "10m_wind_speed": ["10m_wind_speed"],
    "geopotential_height": ["geopotential_height"],
    "temperature": ["temperature"],
    "specific_humidity": ["specific_humidity"],
    "u_wind": ["zonal_wind"],
    "v_wind": ["meridional_wind"],
    "wind_speed": ["wind_speed"],
}

FHR_NAMES = ["fhr", "lead", "lead_time", "forecast_hour", "fhour"]
LEVEL_NAMES = ["level", "lev", "pressure", "pressure_level", "plev", "isobaricInhPa"]


SURFACE_ROWS = [
    ("2m_temperature", None, "2m Temperature", "K"),
    ("2m_specific_humidity", None, "2m Specific Humidity", "kg kg$^{-1}$"),
    ("surface_pressure", None, "Surface Pressure", "hPa"),
    ("10m_u_wind", None, "10m U Wind", "m s$^{-1}$"),
    ("10m_v_wind", None, "10m V Wind", "m s$^{-1}$"),
    ("10m_wind_speed", None, "10m Wind Speed", "m s$^{-1}$"),
]

UPPER_LEVELS = [250, 500, 850]

UPPER_COLUMNS = [
    ("geopotential_height", "Geopotential Height", "m"),
    ("wind_speed", "Wind Speed", "m s$^{-1}$"),
    ("temperature", "Temperature", "K"),
    ("specific_humidity", "Specific Humidity", "g kg$^{-1}$"),
]



def open_dataset(path):
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


def find_name(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def find_var(ds, var_key):
    for name in VAR_ALIASES.get(var_key, [var_key]):
        if name in ds.data_vars:
            return name
    return None


def get_files(model_info, metric):
    files = []
    for pattern in model_info["patterns"]:
        files.extend(sorted(model_info["dir"].glob(pattern.format(metric=metric))))

    conus_files = [f for f in files if ".conus." in f.name]
    if conus_files:
        files = conus_files

    return sorted(set(files))


def normalize_fhr(da):
    fhr_name = find_name(list(da.coords) + list(da.dims), FHR_NAMES)
    if fhr_name is None:
        return None
    if fhr_name != "fhr":
        da = da.rename({fhr_name: "fhr"})
    return da


def normalize_level(da):
    lev_name = find_name(list(da.coords) + list(da.dims), LEVEL_NAMES)
    if lev_name is None:
        return da, None
    if lev_name != "level":
        da = da.rename({lev_name: "level"})
    return da, "level"


def select_level(da, level):
    da, lev_name = normalize_level(da)

    if level is None:
        if lev_name is not None and "level" in da.dims:
            return None
        return da

    if lev_name is None:
        return None

    have = [int(x) for x in da["level"].values]
    if int(level) not in have:
        return None

    return da.sel(level=int(level))


def scale_values(var_key, values):
    arr = np.asarray(values, dtype=float)

    if var_key == "specific_humidity":
        arr = arr * 1000.0

    # Always report surface pressure in hPa.
    # If eagle-tools output is in Pa, typical RMSE/Bias values are usually > 2000.
    # If already in hPa, values usually remain below that range and are not changed.
    if var_key == "surface_pressure":
        finite = arr[np.isfinite(arr)]
        if finite.size:
            median_abs = np.nanmedian(np.abs(finite))
            if np.isfinite(median_abs) and median_abs > 2000.0:
                arr = arr / 100.0

    return arr


def read_series(model_name, metric, var_key, level):
    model_info = MODELS[model_name]
    files = get_files(model_info, metric)

    if not files:
        print(f"WARNING: no {metric} files for {model_name}")
        return np.full(len(FHRS), np.nan)

    arrays = []

    for path in files:
        try:
            ds = open_dataset(path)
        except Exception as e:
            print(f"WARNING: could not open {path}: {e}")
            continue

        var_name = find_var(ds, var_key)
        if var_name is None:
            ds.close()
            continue

        da = ds[var_name]
        da = normalize_fhr(da)

        if da is None or "fhr" not in da.coords:
            ds.close()
            continue

        da = select_level(da, level)
        if da is None:
            ds.close()
            continue

        have_fhrs = [int(x) for x in da["fhr"].values]
        use_fhrs = [f for f in FHRS if f in have_fhrs]

        if not use_fhrs:
            ds.close()
            continue

        da = da.sel(fhr=use_fhrs).reindex(fhr=FHRS)

        if "t0" not in da.dims:
            da = da.expand_dims("t0")

        arrays.append(da.load())
        ds.close()

    if not arrays:
        print(f"WARNING: no usable data for {model_name} {metric} {var_key} {level}")
        return np.full(len(FHRS), np.nan)

    all_da = xr.concat(
        arrays,
        dim="t0_all",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="drop_conflicts",
    )

    mean_dims = [d for d in all_da.dims if d != "fhr"]
    mean = all_da.mean(mean_dims, skipna=True).reindex(fhr=FHRS)

    vals = np.asarray(mean.values, dtype=float).squeeze()
    if vals.shape != (len(FHRS),):
        vals = np.full(len(FHRS), np.nan)

    return scale_values(var_key, vals)


def nice_ylim(metric, data_by_model):
    vals = []
    for arr in data_by_model.values():
        arr = np.asarray(arr, dtype=float)
        vals.extend(arr[np.isfinite(arr)])

    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return None

    if metric == "bias":
        m = np.nanmax(np.abs(vals))
        if not np.isfinite(m) or m == 0:
            m = 1.0
        return (-1.15 * m, 1.15 * m)

    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        hi = lo + 1.0
    pad = 0.10 * (hi - lo)
    return (max(0.0, lo - pad), hi + pad)


def plot_rows(metric, rows, output_name, title, ncols=3):
    nrows = int(np.ceil(len(rows) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, max(4.5, 3.2 * nrows)),
        sharex=True,
    )

    axes = np.atleast_1d(axes).ravel()
    x = np.asarray(FHRS, dtype=float)

    for ax, row in zip(axes, rows):
        var_key, level, label, unit = row

        data_by_model = {}
        for model_name in MODELS:
            data_by_model[model_name] = read_series(model_name, metric, var_key, level)

        for model_name, vals in data_by_model.items():
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.8,
                markersize=3.2,
                label=model_name,
                color=MODEL_COLORS[model_name],
            )

        if metric == "bias":
            ax.axhline(0, color="black", linewidth=0.8)

        ylim = nice_ylim(metric, data_by_model)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xticks(XTICK_FHRS)
        ax.set_xticklabels(XTICK_LABELS, fontsize=9)

    for ax in axes[len(rows):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODELS),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.035,
        "CONUS verification | D1-D10 | 6-hourly lead times | conventional-observation metrics | lower RMSE is better; bias closer to zero is better",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0.03, 0.06, 0.98, 0.90])

    out = OUTPUT_DIR / output_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")



def plot_upper_exact_layout(metric, output_name, title):
    fig, axes = plt.subplots(
        nrows=len(UPPER_LEVELS),
        ncols=len(UPPER_COLUMNS),
        figsize=(18, 10.5),
        sharex=True,
    )

    x = np.asarray(FHRS, dtype=float)

    for i, level in enumerate(UPPER_LEVELS):
        for j, (var_key, var_label, unit) in enumerate(UPPER_COLUMNS):
            ax = axes[i, j]

            data_by_model = {}
            for model_name in MODELS:
                data_by_model[model_name] = read_series(model_name, metric, var_key, level)

            for model_name, vals in data_by_model.items():
                ax.plot(
                    x,
                    vals,
                    linewidth=1.8,
                    label=MODEL_LABELS.get(model_name, model_name),
                    color=MODEL_COLORS[model_name],
                )

            if metric == "bias":
                ax.axhline(0, color="black", linewidth=0.8)

            ylim = nice_ylim(metric, data_by_model)
            if ylim is not None:
                ax.set_ylim(*ylim)

            if i == 0:
                ax.set_title(var_label, fontsize=13)

            if j == 0:
                ax.set_ylabel(f"{level} hPa\n{metric.upper()} ({unit})", fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.grid(False)
            ax.set_xticks(XTICK_FHRS)
            ax.set_xticklabels(XTICK_LABELS, fontsize=10)

            if i == len(UPPER_LEVELS) - 1:
                ax.set_xlabel("Lead time (days)", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODELS),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.925),
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.990)

    fig.text(
        0.5,
        0.948,
        "Nested-EAGLE LAM and HRRR are ~6 km; GFS, AIGFS, and ECMWF IFS are CONUS-subset global models | 6-hourly D1-D10",
        ha="center",
        fontsize=12,
    )

    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.875])

    out = OUTPUT_DIR / output_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")



def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    plot_rows(
        metric="rmse",
        rows=SURFACE_ROWS,
        output_name="rmse_surface_conus_nested_lam_hrrr_gfs_aigfs_ecmwf_D1_D10_6hourly.png",
        title="Surface RMSE: Nested-EAGLE-LAM, HRRR, GFS, AIGFS, and ECMWF IFS",
        ncols=3,
    )

    plot_rows(
        metric="bias",
        rows=SURFACE_ROWS,
        output_name="bias_surface_conus_nested_lam_hrrr_gfs_aigfs_ecmwf_D1_D10_6hourly.png",
        title="Surface Bias: Nested-EAGLE-LAM, HRRR, GFS, AIGFS, and ECMWF IFS",
        ncols=3,
    )

    plot_upper_exact_layout(
        metric="rmse",
        output_name="rmse_upper_conus_nested_lam_hrrr_gfs_aigfs_ecmwf_D1_D10_250_500_850_6hourly.png",
        title="Upper-Air RMSE vs Conventional Obs",
    )

    plot_upper_exact_layout(
        metric="bias",
        output_name="bias_upper_conus_nested_lam_hrrr_gfs_aigfs_ecmwf_D1_D10_250_500_850_6hourly.png",
        title="Upper-Air Bias vs Conventional Obs",
    )


if __name__ == "__main__":
    main()
