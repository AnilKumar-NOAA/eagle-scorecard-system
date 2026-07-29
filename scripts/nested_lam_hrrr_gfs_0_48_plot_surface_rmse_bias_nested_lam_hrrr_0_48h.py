#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Shared scorecard_system output directory.
import os as _scorecard_os
from pathlib import Path as _scorecard_Path
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    "/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard_system/data_system/outputs"
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



BASE = Path("/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard_system/data")
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

METRICS_TO_RUN = [sys.argv[1].lower()] if len(sys.argv) > 1 else ["rmse", "bias"]
for _m in METRICS_TO_RUN:
    if _m not in {"rmse", "bias"}:
        raise SystemExit("Use metric argument: rmse or bias")
METRIC = METRICS_TO_RUN[0]

FHRS = np.arange(0, 49, 6)

MODELS = {
    "Nested-EAGLE LAM": {
        "dir": BASE / "nested_eagle_lam_2025",
        "pattern": f"{METRIC}.convobs.nested-lam.2025-*_to_2025-*.nc",
        "color": "blue",
        "lw": 1.6,
    },
    "HRRR": {
        "dir": BASE / "hrrr_2025",
        "pattern": f"{METRIC}.convobs.lam*.2025-*_to_2025-*.nc",
        "color": "orange",
        "lw": 1.6,
    },
}

MODEL_ORDER = ["Nested-EAGLE LAM", "HRRR"]

SURFACE_VARS = [
    "2m_temperature",
    "2m_specific_humidity",
    "surface_pressure",
    "10m_zonal_wind",
    "10m_meridional_wind",
    "10m_wind_speed",
]

VAR_TITLES = {
    "2m_temperature": "2m Temperature",
    "2m_specific_humidity": "2m Specific Humidity",
    "surface_pressure": "Surface Pressure",
    "10m_zonal_wind": "10m U Wind",
    "10m_meridional_wind": "10m V Wind",
    "10m_wind_speed": "10m Wind Speed",
}

VAR_YLABELS_RMSE = {
    "2m_temperature": "RMSE (K)",
    "2m_specific_humidity": "RMSE (kg kg$^{-1}$)",
    "surface_pressure": "RMSE (hPa)",
    "10m_zonal_wind": "RMSE (m s$^{-1}$)",
    "10m_meridional_wind": "RMSE (m s$^{-1}$)",
    "10m_wind_speed": "RMSE (m s$^{-1}$)",
}

VAR_YLABELS_BIAS = {
    "2m_temperature": "Bias (K)",
    "2m_specific_humidity": "Bias (kg kg$^{-1}$)",
    "surface_pressure": "Bias (hPa)",
    "10m_zonal_wind": "Bias (m s$^{-1}$)",
    "10m_meridional_wind": "Bias (m s$^{-1}$)",
    "10m_wind_speed": "Bias (m s$^{-1}$)",
}


def metric_title():
    return "Surface RMSE vs Conventional Obs" if METRIC == "rmse" else "Surface Bias vs Conventional Obs"


def open_ds(path):
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


def convert_units(varname, da):
    if varname == "surface_pressure":
        vals = np.asarray(da.values, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmedian(np.abs(finite)) > 20:
            da = da / 100.0
    return da


def load_model_series(model_name, varname):
    info = MODELS[model_name]
    files = sorted(info["dir"].glob(info["pattern"]))

    print(f"{model_name:18s} {varname:24s} files={len(files)}")

    arrays = []
    opened = []

    for f in files:
        ds = open_ds(f)
        opened.append(ds)

        if varname not in ds.data_vars:
            continue

        da = ds[varname]

        if "fhr" not in da.coords:
            continue

        if "level" in da.dims:
            continue

        have_fhr = [int(x) for x in da["fhr"].values]
        use_fhr = [int(x) for x in FHRS if int(x) in have_fhr]

        if not use_fhr:
            continue

        da = da.sel(fhr=use_fhr)
        da = convert_units(varname, da)

        if "t0" not in da.dims:
            da = da.expand_dims("t0")

        arrays.append(da.load())

    if not arrays:
        for ds in opened:
            ds.close()
        return None, None

    all_da = xr.concat(arrays, dim="t0")

    mean = all_da.mean("t0", skipna=True)
    std = all_da.std("t0", skipna=True)
    n = all_da.sizes.get("t0", 1)
    spread = std / np.sqrt(max(n, 1))

    for ds in opened:
        ds.close()

    return mean, spread


def collect_data():
    data = {}
    for var in SURFACE_VARS:
        data[var] = {}
        for model in MODEL_ORDER:
            mean, spread = load_model_series(model, var)
            if mean is not None:
                data[var][model] = {
                    "mean": mean,
                    "spread": spread,
                    "color": MODELS[model]["color"],
                    "lw": MODELS[model]["lw"],
                }
    return data


def set_symmetric_ylim(ax, plotted_values, pad=0.10):
    vals = []
    for arr in plotted_values:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)

    if not vals:
        return

    vmax = max(np.nanmax(np.abs(v)) for v in vals)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    vmax *= 1.0 + pad
    ax.set_ylim(-vmax, vmax)


def style_axis(ax):
    ax.set_xlim(-1, 49)
    ax.set_xticks(FHRS)
    ax.set_xticklabels([str(int(x)) for x in FHRS])
    ax.tick_params(axis="both", labelsize=9, width=1.0, length=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def plot_panel(ax, data, varname):
    handles = {}
    plotted_values = []

    for model in MODEL_ORDER:
        if model not in data.get(varname, {}):
            continue

        mean = data[varname][model]["mean"]
        spread = data[varname][model]["spread"]
        color = data[varname][model]["color"]
        lw = data[varname][model]["lw"]

        x = mean["fhr"].values.astype(float)
        y = np.asarray(mean.values, dtype=float)
        e = np.asarray(spread.values, dtype=float)

        line, = ax.plot(
            x,
            y,
            color=color,
            linewidth=lw,
            label=model,
            solid_capstyle="round",
        )

        ax.fill_between(
            x,
            y - e,
            y + e,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

        handles[model] = line
        plotted_values.append(y)

    style_axis(ax)

    if METRIC == "bias":
        set_symmetric_ylim(ax, plotted_values)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)

    return handles


def run_one_metric(metric):
    global METRIC
    METRIC = metric
    data = collect_data()

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharex=True)
    axes = axes.ravel()

    legend_handles = {}
    ylabels = VAR_YLABELS_RMSE if METRIC == "rmse" else VAR_YLABELS_BIAS

    for i, varname in enumerate(SURFACE_VARS):
        ax = axes[i]
        handles = plot_panel(ax, data, varname)

        for k, v in handles.items():
            legend_handles.setdefault(k, v)

        ax.set_title(VAR_TITLES[varname], fontsize=12, pad=6)
        ax.set_ylabel(ylabels[varname], fontsize=10, labelpad=8)

        if i >= 4:
            ax.set_xlabel("Lead Time (hours)", fontsize=10)

    fig.suptitle(metric_title(), fontsize=18, fontweight="bold", y=0.995)

    fig.text(
        0.5,
        0.955,
        "Nested-EAGLE LAM vs HRRR | CONUS | 0-48 h lead time | Both models ~6 km resolution",
        ha="center",
        va="center",
        fontsize=12,
    )

    if legend_handles:
        fig.legend(
            [legend_handles[m] for m in MODEL_ORDER if m in legend_handles],
            [m for m in MODEL_ORDER if m in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            frameon=False,
            fontsize=10,
            handlelength=1.8,
            labelspacing=0.25,
            handletextpad=0.45,
            ncol=2,
            columnspacing=1.5,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=[0.05, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{METRIC}_surface_conus_nested_lam_hrrr_0_48h.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


def main():
    for metric in METRICS_TO_RUN:
        print("\n============================================================")
        print(f"Creating {metric.upper()} surface plot")
        print("============================================================")
        run_one_metric(metric)


if __name__ == "__main__":
    main()
