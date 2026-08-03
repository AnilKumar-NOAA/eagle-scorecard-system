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
    str(Path(__file__).resolve().parents[1] / "outputs")
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# Paths/settings
# ============================================================

BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "data/new_data"))
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

# Run:
#   python plot_upper_rmse_bias_nested_lam_hrrr_0_48h.py rmse
#   python plot_upper_rmse_bias_nested_lam_hrrr_0_48h.py bias
METRIC = sys.argv[1].lower() if len(sys.argv) > 1 else "rmse"
if METRIC not in {"rmse", "bias"}:
    raise SystemExit("Use metric argument: rmse or bias")

# 0-48 h only, 6-hourly interval
FHRS = np.arange(0, 49, 6)
LEVELS = [250, 500, 850]

MODELS = {
    "Nested-EAGLE LAM": {
        "dir": BASE / "nested_eagle_lam_2025",
        "pattern": f"{METRIC}.convobs.nested-lam.nc",
        "color": "blue",
        "lw": 1.6,
    },
    "HRRR": {
        "dir": BASE / "hrrr_2025",
        "pattern": f"{METRIC}.convobs.lam.nc",
        "color": "orange",
        "lw": 1.6,
    },
}

MODEL_ORDER = ["Nested-EAGLE LAM", "HRRR"]

UPPER_VARS = [
    "geopotential_height",
    "wind_speed",
    "temperature",
    "specific_humidity",
]

VAR_TITLES = {
    "geopotential_height": "Geopotential Height",
    "wind_speed": "Wind Speed",
    "temperature": "Temperature",
    "specific_humidity": "Specific Humidity",
}

VAR_YLABELS_RMSE = {
    "geopotential_height": "RMSE (m)",
    "wind_speed": "RMSE (m s$^{-1}$)",
    "temperature": "RMSE (K)",
    "specific_humidity": "RMSE (kg kg$^{-1}$)",
}

VAR_YLABELS_BIAS = {
    "geopotential_height": "Bias (m)",
    "wind_speed": "Bias (m s$^{-1}$)",
    "temperature": "Bias (K)",
    "specific_humidity": "Bias (kg kg$^{-1}$)",
}


# ============================================================
# Helpers
# ============================================================

def metric_title():
    if METRIC == "rmse":
        return "Upper-Air RMSE vs Conventional Obs"
    return "Upper-Air Bias vs Conventional Obs"


def open_ds(path):
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


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

        if "fhr" not in da.coords or "level" not in da.coords:
            continue

        have_fhr = [int(x) for x in da["fhr"].values]
        use_fhr = [int(x) for x in FHRS if int(x) in have_fhr]

        have_level = [int(x) for x in da["level"].values]
        use_level = [int(x) for x in LEVELS if int(x) in have_level]

        if not use_fhr or not use_level:
            continue

        da = da.sel(fhr=use_fhr, level=use_level)

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

    # Shaded band: standard error across initialization dates.
    spread = std / np.sqrt(max(n, 1))

    for ds in opened:
        ds.close()

    return mean, spread


def collect_data():
    data = {}

    for var in UPPER_VARS:
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


def plot_panel(ax, data, varname, level):
    handles = {}
    plotted_values = []

    for model in MODEL_ORDER:
        if model not in data.get(varname, {}):
            continue

        mean = data[varname][model]["mean"]
        spread = data[varname][model]["spread"]
        color = data[varname][model]["color"]
        lw = data[varname][model]["lw"]

        if level not in [int(x) for x in mean["level"].values]:
            continue

        mean_lev = mean.sel(level=level)
        spread_lev = spread.sel(level=level)

        x = mean_lev["fhr"].values.astype(float)
        y = np.asarray(mean_lev.values, dtype=float)
        e = np.asarray(spread_lev.values, dtype=float)

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
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )

    return handles


# ============================================================
# Main
# ============================================================

def main():
    data = collect_data()

    fig, axes = plt.subplots(
        len(LEVELS),
        len(UPPER_VARS),
        figsize=(13.5, 8.2),
        sharex=True,
    )

    legend_handles = {}
    ylabels = VAR_YLABELS_RMSE if METRIC == "rmse" else VAR_YLABELS_BIAS

    for i, level in enumerate(LEVELS):
        for j, varname in enumerate(UPPER_VARS):
            ax = axes[i, j]
            handles = plot_panel(ax, data, varname, level)

            for k, v in handles.items():
                legend_handles.setdefault(k, v)

            if i == 0:
                ax.set_title(VAR_TITLES[varname], fontsize=12, pad=6)

            if j == 0:
                ax.set_ylabel(ylabels[varname], fontsize=10, labelpad=8)
                ax.text(
                    -0.33,
                    0.50,
                    f"{level} hPa",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

            if i == len(LEVELS) - 1:
                ax.set_xlabel("Lead Time (hours)", fontsize=10)

    fig.suptitle(
        metric_title(),
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

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

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{METRIC}_upper_conus_nested_lam_hrrr_0_48h_250_500_850.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


if __name__ == "__main__":
    main()
