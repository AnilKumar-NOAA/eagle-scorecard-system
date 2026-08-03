#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


DERIVED_SURFACE_SKIP_VARS = {"surface_pressure", "2m_specific_humidity"}
DERIVED_SURFACE_SKIP_MODEL_TOKENS = ("AIGFS", "AIFS")

def skip_aigfs_aifs_derived_surface(model_name, varname, level=None):
    return (
        level is None
        and varname in DERIVED_SURFACE_SKIP_VARS
        and any(token in str(model_name) for token in DERIVED_SURFACE_SKIP_MODEL_TOKENS)
    )



# ============================================================
# Paths/settings
# ============================================================

BASE = Path(__file__).resolve().parents[1]
DATA_BASE = BASE / "data/new_data"
OUTDIR = BASE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

FHRS = np.arange(0, 241, 6)
LEVELS = [250, 500, 850]
METRICS = ["rmse", "bias"]


MODELS = {
    "Nested-EAGLE LAM (~6 km)": {
        "dir": DATA_BASE / "nested_eagle_lam_2025",
        "pattern": "{metric}.convobs.nested-lam.nc",
        "color": "blue",
        "lw": 1.4,
    },
    "HRRR (~6 km)": {
        "dir": DATA_BASE / "hrrr_2025",
        "pattern": "{metric}.convobs.lam.nc",
        "color": "orange",
        "lw": 1.4,
    },
    "GFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "gfs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "black",
        "lw": 1.4,
    },
    "AIGFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "aigfs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "green",
        "lw": 1.4,
    },
    "AIFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "aifs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "red",
        "lw": 1.4,
    },
}

MODEL_ORDER = list(MODELS.keys())


SURFACE_VARS = [
    "2m_temperature",
    "2m_specific_humidity",
    "surface_pressure",
    "10m_zonal_wind",
    "10m_meridional_wind",
    "10m_wind_speed",
]

UPPER_VARS = [
    "geopotential_height",
    "wind_speed",
    "temperature",
    "specific_humidity",
]

VAR_TITLES = {
    "2m_temperature": "2m Temperature",
    "2m_specific_humidity": "2m Specific Humidity",
    "surface_pressure": "Surface Pressure",
    "10m_zonal_wind": "10m U Wind",
    "10m_meridional_wind": "10m V Wind",
    "10m_wind_speed": "10m Wind Speed",
    "geopotential_height": "Geopotential Height",
    "wind_speed": "Wind Speed",
    "temperature": "Temperature",
    "specific_humidity": "Specific Humidity",
}

SURFACE_YLABELS_RMSE = {
    "2m_temperature": "RMSE (K)",
    "2m_specific_humidity": "RMSE (kg kg$^{-1}$)",
    "surface_pressure": "RMSE (hPa)",
    "10m_zonal_wind": "RMSE (m s$^{-1}$)",
    "10m_meridional_wind": "RMSE (m s$^{-1}$)",
    "10m_wind_speed": "RMSE (m s$^{-1}$)",
}

SURFACE_YLABELS_BIAS = {
    "2m_temperature": "Bias (K)",
    "2m_specific_humidity": "Bias (kg kg$^{-1}$)",
    "surface_pressure": "Bias (hPa)",
    "10m_zonal_wind": "Bias (m s$^{-1}$)",
    "10m_meridional_wind": "Bias (m s$^{-1}$)",
    "10m_wind_speed": "Bias (m s$^{-1}$)",
}

UPPER_YLABELS_RMSE = {
    "geopotential_height": "RMSE (m)",
    "wind_speed": "RMSE (m s$^{-1}$)",
    "temperature": "RMSE (K)",
    "specific_humidity": "RMSE (kg kg$^{-1}$)",
}

UPPER_YLABELS_BIAS = {
    "geopotential_height": "Bias (m)",
    "wind_speed": "Bias (m s$^{-1}$)",
    "temperature": "Bias (K)",
    "specific_humidity": "Bias (kg kg$^{-1}$)",
}


# ============================================================
# Helpers
# ============================================================

def open_ds(path):
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


def convert_units(varname, da):
    if varname == "surface_pressure":
        vals = np.asarray(da.values, dtype=float)
        finite = vals[np.isfinite(vals)]

        units = str(da.attrs.get("units", "")).lower()

        if "pa" in units and "hpa" not in units:
            da = da / 100.0
        elif finite.size:
            # Fallback: if the pressure metric looks like Pa, convert to hPa.
            # This keeps all models on hPa for both RMSE and Bias.
            max_abs = np.nanmax(np.abs(finite))
            median_abs = np.nanmedian(np.abs(finite))
            if max_abs > 200.0 or median_abs > 20.0:
                da = da / 100.0

    return da


def metric_title(metric, field_type):
    m = "RMSE" if metric == "rmse" else "Bias"
    if field_type == "surface":
        return f"Surface {m} vs Conventional Obs"
    return f"Upper-Air {m} vs Conventional Obs"


def load_model_series(metric, model_name, varname, levels=None):
    if skip_aigfs_aifs_derived_surface(model_name, varname, levels):
        print(f"{metric.upper():4s} | {model_name:30s} | {varname:24s} | skipped derived surface field")
        return None, None

    info = MODELS[model_name]
    files = sorted(info["dir"].glob(info["pattern"].format(metric=metric)))

    # Prefer CONUS-specific global files if available.
    conus_files = [f for f in files if ".conus." in f.name]
    if conus_files:
        files = conus_files

    print(f"{metric.upper():4s} | {model_name:30s} | {varname:24s} | files={len(files)}")

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

        have_fhr = [int(x) for x in da["fhr"].values]
        use_fhr = [int(x) for x in FHRS if int(x) in have_fhr]

        if not use_fhr:
            continue

        da = da.sel(fhr=use_fhr)

        if levels is None:
            if "level" in da.dims:
                continue
            da = convert_units(varname, da)
        else:
            if "level" not in da.coords:
                continue

            have_level = [int(x) for x in da["level"].values]
            use_level = [int(x) for x in levels if int(x) in have_level]

            if not use_level:
                continue

            da = da.sel(level=use_level)

        da = da.reindex(fhr=FHRS)

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


def collect_surface_data(metric):
    data = {}

    for var in SURFACE_VARS:
        data[var] = {}
        for model in MODEL_ORDER:
            mean, spread = load_model_series(metric, model, var, levels=None)
            if mean is not None:
                data[var][model] = {
                    "mean": mean,
                    "spread": spread,
                    "color": MODELS[model]["color"],
                    "lw": MODELS[model]["lw"],
                }

    return data


def collect_upper_data(metric):
    data = {}

    for var in UPPER_VARS:
        data[var] = {}
        for model in MODEL_ORDER:
            mean, spread = load_model_series(metric, model, var, levels=LEVELS)
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
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.tick_params(axis="both", labelsize=9, width=1.0, length=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def plot_surface_panel(ax, data, metric, varname):
    handles = {}
    plotted_values = []

    for model in MODEL_ORDER:
        if model not in data.get(varname, {}):
            continue

        mean = data[varname][model]["mean"]
        spread = data[varname][model]["spread"]
        color = data[varname][model]["color"]
        lw = data[varname][model]["lw"]

        x = mean["fhr"].values.astype(float) / 24.0
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
            alpha=0.12,
            linewidth=0,
        )

        handles[model] = line
        plotted_values.append(y)

    style_axis(ax)

    if metric == "bias":
        set_symmetric_ylim(ax, plotted_values)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.85)

    return handles


def plot_upper_panel(ax, data, metric, varname, level):
    handles = {}
    plotted_values = []

    for model in MODEL_ORDER:
        if model not in data.get(varname, {}):
            continue

        mean = data[varname][model]["mean"]
        spread = data[varname][model]["spread"]
        color = data[varname][model]["color"]
        lw = data[varname][model]["lw"]

        if "level" not in mean.coords:
            continue

        if level not in [int(x) for x in mean["level"].values]:
            continue

        mean_lev = mean.sel(level=level)
        spread_lev = spread.sel(level=level)

        x = mean_lev["fhr"].values.astype(float) / 24.0
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
            alpha=0.12,
            linewidth=0,
        )

        handles[model] = line
        plotted_values.append(y)

    style_axis(ax)

    if metric == "bias":
        set_symmetric_ylim(ax, plotted_values)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.85)

    return handles


# ============================================================
# Plot drivers
# ============================================================

def make_surface_plot(metric):
    data = collect_surface_data(metric)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.2, 9.2),
        sharex=True,
    )

    axes = axes.ravel()
    legend_handles = {}
    ylabels = SURFACE_YLABELS_RMSE if metric == "rmse" else SURFACE_YLABELS_BIAS

    for i, varname in enumerate(SURFACE_VARS):
        ax = axes[i]
        handles = plot_surface_panel(ax, data, metric, varname)

        for k, v in handles.items():
            legend_handles.setdefault(k, v)

        ax.set_title(VAR_TITLES[varname], fontsize=12, pad=6)
        ax.set_ylabel(ylabels[varname], fontsize=10, labelpad=8)

        if i >= 4:
            ax.set_xlabel("Lead time (days)", fontsize=10)

    fig.suptitle(
        metric_title(metric, "surface"),
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.955,
        "Nested-EAGLE LAM and HRRR are ~6 km; GFS, AIGFS, and AIFS are CONUS subsets at 0.25 degree | 6-hourly D1-D10",
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
            ncol=5,
            columnspacing=1.1,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=[0.05, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{metric}_surface_conus_nested_lam_hrrr_gfs_aigfs_aifs_D1_D10_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


def make_upper_plot(metric):
    data = collect_upper_data(metric)

    fig, axes = plt.subplots(
        len(LEVELS),
        len(UPPER_VARS),
        figsize=(13.5, 8.8),
        sharex=True,
    )

    legend_handles = {}
    ylabels = UPPER_YLABELS_RMSE if metric == "rmse" else UPPER_YLABELS_BIAS

    for i, level in enumerate(LEVELS):
        for j, varname in enumerate(UPPER_VARS):
            ax = axes[i, j]
            handles = plot_upper_panel(ax, data, metric, varname, level)

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
                ax.set_xlabel("Lead time (days)", fontsize=10)

    fig.suptitle(
        metric_title(metric, "upper"),
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.955,
        "Nested-EAGLE LAM and HRRR are ~6 km; GFS, AIGFS, and AIFS are CONUS subsets at 0.25 degree | 6-hourly D1-D10",
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
            ncol=5,
            columnspacing=1.1,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{metric}_upper_conus_nested_lam_hrrr_gfs_aigfs_aifs_D1_D10_250_500_850_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


def main():
    for metric in METRICS:
        print("\n============================================================")
        print(f"Creating {metric.upper()} plots")
        print("============================================================")
        make_surface_plot(metric)
        make_upper_plot(metric)

    print("\nDone.")


if __name__ == "__main__":
    main()
