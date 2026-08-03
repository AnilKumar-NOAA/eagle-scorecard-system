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



BASE = Path(__file__).resolve().parents[1]
DATA_BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", BASE / "data/new_data"))
OUTDIR = Path(__import__("os").environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", BASE / "outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)

FHRS = np.arange(0, 241, 24)
LEVELS = [250, 500, 850]
METRICS = ["rmse", "bias"]

ANCHOR_MODEL = "GFS CONUS (0.25 deg)"

MODELS = {
    "Nested-EAGLE LAM (~6 km)": {
        "dir": DATA_BASE / "nested_eagle_lam_2025",
        "pattern": "{metric}.convobs.nested-lam.nc",
        "color": "blue",
        "lw": 1.5,
    },
    "HRRR (~6 km)": {
        "dir": DATA_BASE / "hrrr_2025",
        "pattern": "{metric}.convobs.lam.nc",
        "color": "orange",
        "lw": 1.5,
    },
    "GFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "gfs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "black",
        "lw": 1.5,
    },
    "AIGFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "aigfs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "green",
        "lw": 1.5,
    },
    "AIFS CONUS (0.25 deg)": {
        "dir": DATA_BASE / "aifs_2025",
        "pattern": "{metric}.convobs.global.conus.nc",
        "color": "red",
        "lw": 1.5,
    },
}

MODEL_ORDER = [
    "Nested-EAGLE LAM (~6 km)",
    "HRRR (~6 km)",
    "AIGFS CONUS (0.25 deg)",
    "AIFS CONUS (0.25 deg)",
]

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

SURFACE_UNITS = {
    "2m_temperature": "K",
    "2m_specific_humidity": "kg kg$^{-1}$",
    "surface_pressure": "hPa",
    "10m_zonal_wind": "m s$^{-1}$",
    "10m_meridional_wind": "m s$^{-1}$",
    "10m_wind_speed": "m s$^{-1}$",
}

UPPER_UNITS = {
    "geopotential_height": "m",
    "wind_speed": "m s$^{-1}$",
    "temperature": "K",
    "specific_humidity": "kg kg$^{-1}$",
}


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
            max_abs = np.nanmax(np.abs(finite))
            median_abs = np.nanmedian(np.abs(finite))
            if max_abs > 200.0 or median_abs > 20.0:
                da = da / 100.0

    return da


def load_model_series(metric, model_name, varname, levels=None):
    if skip_aigfs_aifs_derived_surface(model_name, varname, levels):
        print(f"{metric.upper():4s} | {model_name:30s} | {varname:24s} | skipped derived surface field")
        return None

    info = MODELS[model_name]
    files = sorted(info["dir"].glob(info["pattern"].format(metric=metric)))

    conus_files = [f for f in files if ".conus." in f.name]
    if conus_files:
        files = conus_files

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
        return None

    all_da = xr.concat(arrays, dim="t0")
    mean = all_da.mean("t0", skipna=True)

    for ds in opened:
        ds.close()

    return mean


def style_axis(ax):
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.tick_params(axis="both", labelsize=9, width=1.0, length=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


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

    ax.set_ylim(-vmax * (1.0 + pad), vmax * (1.0 + pad))


def plot_surface_difference_panel(ax, metric, varname):
    handles = {}
    plotted_values = []

    gfs = load_model_series(metric, ANCHOR_MODEL, varname, levels=None)
    if gfs is None:
        style_axis(ax)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        return handles

    for model in MODEL_ORDER:
        model_da = load_model_series(metric, model, varname, levels=None)
        if model_da is None:
            continue

        diff = model_da - gfs

        x = diff["fhr"].values.astype(float) / 24.0
        y = np.asarray(diff.values, dtype=float)

        line, = ax.plot(
            x,
            y,
            color=MODELS[model]["color"],
            linewidth=MODELS[model]["lw"],
            label=model,
            solid_capstyle="round",
        )

        handles[model] = line
        plotted_values.append(y)

    style_axis(ax)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.85)
    set_symmetric_ylim(ax, plotted_values)

    return handles


def plot_upper_difference_panel(ax, metric, varname, level):
    handles = {}
    plotted_values = []

    gfs = load_model_series(metric, ANCHOR_MODEL, varname, levels=LEVELS)
    if gfs is None or "level" not in gfs.coords or level not in [int(x) for x in gfs["level"].values]:
        style_axis(ax)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        return handles

    gfs_lev = gfs.sel(level=level)

    for model in MODEL_ORDER:
        model_da = load_model_series(metric, model, varname, levels=LEVELS)
        if model_da is None:
            continue

        if "level" not in model_da.coords or level not in [int(x) for x in model_da["level"].values]:
            continue

        model_lev = model_da.sel(level=level)
        diff = model_lev - gfs_lev

        x = diff["fhr"].values.astype(float) / 24.0
        y = np.asarray(diff.values, dtype=float)

        line, = ax.plot(
            x,
            y,
            color=MODELS[model]["color"],
            linewidth=MODELS[model]["lw"],
            label=model,
            solid_capstyle="round",
        )

        handles[model] = line
        plotted_values.append(y)

    style_axis(ax)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.85)
    set_symmetric_ylim(ax, plotted_values)

    return handles


def make_surface_difference_plot(metric):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.2, 9.2),
        sharex=True,
    )

    axes = axes.ravel()
    legend_handles = {}

    for i, varname in enumerate(SURFACE_VARS):
        ax = axes[i]
        handles = plot_surface_difference_panel(ax, metric, varname)

        for k, v in handles.items():
            legend_handles.setdefault(k, v)

        ax.set_title(VAR_TITLES[varname], fontsize=12, pad=6)
        ax.set_ylabel(f"{metric.upper()} - GFS ({SURFACE_UNITS[varname]})", fontsize=10, labelpad=8)

        if i >= 4:
            ax.set_xlabel("Lead time (days)", fontsize=10)

    m = "RMSE" if metric == "rmse" else "Bias"

    fig.suptitle(
        f"Surface {m} Difference from GFS",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    if metric == "rmse":
        subtitle = "Difference = model RMSE - GFS RMSE | Negative difference means better than GFS | 6-hourly D1-D10"
    else:
        subtitle = "Difference = model bias - GFS bias | Zero line indicates same bias as GFS | 6-hourly D1-D10"

    fig.text(
        0.5,
        0.955,
        subtitle,
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
            ncol=4,
            columnspacing=1.4,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=[0.05, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{metric}_surface_difference_from_gfs_conus_D1_D10_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


def make_upper_difference_plot(metric):
    fig, axes = plt.subplots(
        len(LEVELS),
        len(UPPER_VARS),
        figsize=(13.5, 8.8),
        sharex=True,
    )

    legend_handles = {}

    for i, level in enumerate(LEVELS):
        for j, varname in enumerate(UPPER_VARS):
            ax = axes[i, j]
            handles = plot_upper_difference_panel(ax, metric, varname, level)

            for k, v in handles.items():
                legend_handles.setdefault(k, v)

            if i == 0:
                ax.set_title(VAR_TITLES[varname], fontsize=12, pad=6)

            if j == 0:
                ax.set_ylabel(f"{metric.upper()} - GFS ({UPPER_UNITS[varname]})", fontsize=10, labelpad=8)
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

    m = "RMSE" if metric == "rmse" else "Bias"

    fig.suptitle(
        f"Upper-Air {m} Difference from GFS",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    if metric == "rmse":
        subtitle = "Difference = model RMSE - GFS RMSE | Negative difference means better than GFS | 6-hourly D1-D10"
    else:
        subtitle = "Difference = model bias - GFS bias | Zero line indicates same bias as GFS | 6-hourly D1-D10"

    fig.text(
        0.5,
        0.955,
        subtitle,
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
            ncol=4,
            columnspacing=1.4,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.86])

    outpng = OUTDIR / f"{metric}_upper_difference_from_gfs_conus_D1_D10_250_500_850_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpng)


def main():
    for metric in METRICS:
        print("\n============================================================")
        print(f"Creating {metric.upper()} difference-from-GFS plots")
        print("============================================================")
        make_surface_difference_plot(metric)
        make_upper_difference_plot(metric)

    print("\nDone.")


if __name__ == "__main__":
    main()
