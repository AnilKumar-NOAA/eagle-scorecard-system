#!/usr/bin/env python3

from pathlib import Path
import warnings

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


# ----------------------------------------------------------------------
# scorecard_system output control
# All figures, CSVs, and text products from this script are written here.
# Input data remain under the shared scorecard data directory.
# ----------------------------------------------------------------------
import os as _scorecard_os
from pathlib import Path as _scorecard_Path
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "outputs")
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
SCORECARD = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "data/new_data"))
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_INFO = {
    "Nested EAGLE": {
        "dir": SCORECARD / "nested_eagle_global_2025",
        "pattern": "{metric}.convobs.nested-global.nc",
        "color": "royalblue",
    },
    "GFS": {
        "dir": SCORECARD / "gfs_2025",
        "pattern": "{metric}.convobs.global.nc",
        "color": "black",
    },
    "AIFS": {
        "dir": SCORECARD / "aifs_2025",
        "pattern": "{metric}.convobs.global.nc",
        "color": "crimson",
    },
    "AIGFS": {
        "dir": SCORECARD / "aigfs_2025",
        "pattern": "{metric}.convobs.global.nc",
        "color": "forestgreen",
    },
}

# ============================================================
# Plot settings
# ============================================================
FHR_PLOT = np.arange(0, 241, 24)
X_DAYS = FHR_PLOT / 24.0
XTICKS_MAJOR = np.arange(0, 11, 1)
XTICK_LABELS = [f"D{i}" for i in XTICKS_MAJOR]

LINEWIDTH = 1.5
MARKERSIZE = 3.0

SURFACE_VARS = [
    "2m_temperature",
    "surface_pressure",
    "10m_zonal_wind",
    "10m_meridional_wind",
    "10m_wind_speed",
    "2m_specific_humidity",
]

UPPER_VARS = [
    "geopotential_height",
    "temperature",
    "specific_humidity",
    "zonal_wind",
    "meridional_wind",
    "wind_speed",
]

LEVELS = [250, 500, 850]

VAR_LABELS = {
    "2m_temperature": "T2M (K)",
    "surface_pressure": "SP (hPa)",
    "10m_zonal_wind": "U10 (m/s)",
    "10m_meridional_wind": "V10 (m/s)",
    "10m_wind_speed": "WSPD10 (m/s)",
    "2m_specific_humidity": "Q2 (g/kg)",
    "geopotential_height": "HGT (m)",
    "temperature": "T (K)",
    "specific_humidity": "Q (g/kg)",
    "zonal_wind": "U (m/s)",
    "meridional_wind": "V (m/s)",
    "wind_speed": "WSPD (m/s)",
}

VAR_TITLES = {
    "2m_temperature": "2m Temperature",
    "surface_pressure": "Surface Pressure",
    "10m_zonal_wind": "10m Zonal Wind",
    "10m_meridional_wind": "10m Meridional Wind",
    "10m_wind_speed": "10m Wind Speed",
    "2m_specific_humidity": "2m Specific Humidity",
    "geopotential_height": "Geopotential Height",
    "temperature": "Temperature",
    "specific_humidity": "Specific Humidity",
    "zonal_wind": "Zonal Wind",
    "meridional_wind": "Meridional Wind",
    "wind_speed": "Wind Speed",
}


# ============================================================
# Helpers
# ============================================================
def open_many_mean(files):
    """Open monthly files, concatenate over t0, and return annual mean."""
    ds_list = []
    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4", decode_timedelta=True)
        if "fhr" in ds:
            ds = ds.sortby("fhr").reindex(fhr=FHR_PLOT)
        ds.load()
        ds_list.append(ds)

    if not ds_list:
        return None

    combined = xr.concat(
        ds_list,
        dim="t0",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    mean_ds = combined.mean("t0", skipna=True)

    for ds in ds_list:
        ds.close()

    return mean_ds


def load_metric_means(metric):
    """Load annual mean dataset for each model for one metric."""
    out = {}
    print(f"\nLoading metric: {metric}")
    for model, info in MODEL_INFO.items():
        files = sorted(info["dir"].glob(info["pattern"].format(metric=metric)))
        if not files:
            print(f"  {model:13s}: no files")
            continue

        mean_ds = open_many_mean(files)
        if mean_ds is None:
            print(f"  {model:13s}: no data")
            continue

        out[model] = mean_ds
        print(f"  {model:13s}: loaded ({len(files)} files)")
    return out


def convert_units(varname, data):
    """Apply unit conversions for plotting."""
    if data is None:
        return None

    arr = np.asarray(data, dtype=float)

    # pressure: Pa -> hPa
    if varname == "surface_pressure":
        return arr / 100.0

    # humidity: kg/kg -> g/kg
    if varname in ["specific_humidity", "2m_specific_humidity"]:
        return arr * 1000.0

    # others unchanged
    return arr


def get_surface_series(ds, varname):
    if varname not in ds.data_vars:
        return None
    da = ds[varname]
    if "fhr" not in da.dims:
        return None
    return convert_units(varname, da.values)


def get_upper_series(ds, varname, level):
    if varname not in ds.data_vars:
        return None
    da = ds[varname]
    if "level" not in da.dims or "fhr" not in da.dims:
        return None
    try:
        da = da.sel(level=level)
    except Exception:
        return None
    return convert_units(varname, da.values)


def set_day_axis(ax, show_xlabel=False):
    # Data are plotted every 6 hours, expressed in days.
    ax.set_xlim(0, 10)

    # Major labels are daily D0-D10.
    ax.set_xticks(XTICKS_MAJOR)
    ax.set_xticklabels(XTICK_LABELS, fontsize=8)

    # Minor ticks every 6 hours = 0.25 day.
    ax.set_xticks(np.arange(0, 10.01, 0.25), minor=True)
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)

    if show_xlabel:
        ax.set_xlabel("Lead time (days)", fontsize=10)


def style_axis(ax, metric_key):
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.4)
    ax.tick_params(axis="both", labelsize=8)


def apply_surface_ylim(ax, plotted_vals, metric_key):
    vals = np.asarray(plotted_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    if metric_key == "bias":
        lim = np.nanmax(np.abs(vals))
        if lim == 0:
            lim = 1.0
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
    else:
        vmax = np.nanmax(vals)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ax.set_ylim(0, 1.1 * vmax)


def apply_upper_row_ylim(row_axes, row_vals, metric_key):
    vals = np.asarray(row_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    if metric_key == "bias":
        lim = np.nanmax(np.abs(vals))
        if lim == 0:
            lim = 1.0
        for ax in row_axes:
            ax.set_ylim(-1.1 * lim, 1.1 * lim)
    else:
        vmax = np.nanmax(vals)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        for ax in row_axes:
            ax.set_ylim(0, 1.1 * vmax)


# ============================================================
# Plot functions
# ============================================================
def plot_surface_panel(metric_key, model_means):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    axes = axes.ravel()

    handles = []
    labels = []

    for ax, varname in zip(axes, SURFACE_VARS):
        plotted_any = False
        subplot_vals = []

        for model, ds in model_means.items():
            if skip_aigfs_aifs_derived_surface(model, varname):
                continue

            series = get_surface_series(ds, varname)
            if series is None:
                continue

            ax.plot(
                X_DAYS,
                series,
                color=MODEL_INFO[model]["color"],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=model,
            )

            subplot_vals.extend(series[np.isfinite(series)].tolist())
            plotted_any = True

        if not plotted_any:
            ax.set_visible(False)
            continue

        ax.set_title(VAR_TITLES[varname], fontsize=10, pad=4)
        ax.set_ylabel(VAR_LABELS[varname], fontsize=9)
        style_axis(ax, metric_key)
        apply_surface_ylim(ax, subplot_vals, metric_key)

        if not handles:
            for model in model_means:
                h, = ax.plot([], [], color=MODEL_INFO[model]["color"], linewidth=LINEWIDTH)
                handles.append(h)
                labels.append(model)

    for i, ax in enumerate(axes):
        if ax.get_visible():
            set_day_axis(ax, show_xlabel=(i >= 3))

    metric_title = "RMSE" if metric_key == "rmse" else "Mean Error (Bias)"
    fig.suptitle(f"{metric_title} | GLOBAL", fontsize=13, y=0.94555)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.880),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.3,
    )

    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.900])

    outpng = OUTDIR / f"global_surface_{metric_key}_panel_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {outpng}")


def plot_upper_panel(metric_key, model_means):
    nrows = len(UPPER_VARS)
    ncols = len(LEVELS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 14), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    handles = []
    labels = []

    for j, lev in enumerate(LEVELS):
        axes[0, j].set_title(f"{lev} hPa", fontsize=10, pad=4)

    for i, varname in enumerate(UPPER_VARS):
        row_vals = []

        for j, lev in enumerate(LEVELS):
            ax = axes[i, j]
            plotted_any = False

            for model, ds in model_means.items():
                series = get_upper_series(ds, varname, lev)
                if series is None:
                    continue

                ax.plot(
                    X_DAYS,
                    series,
                    color=MODEL_INFO[model]["color"],
                    linewidth=LINEWIDTH,
                    markersize=MARKERSIZE,
                    label=model,
                )

                row_vals.extend(series[np.isfinite(series)].tolist())
                plotted_any = True

            if not plotted_any:
                ax.set_visible(False)
                continue

            style_axis(ax, metric_key)

            if j == 0:
                ax.set_ylabel(VAR_LABELS[varname], fontsize=9)

            if not handles:
                for model in model_means:
                    h, = ax.plot([], [], color=MODEL_INFO[model]["color"], linewidth=LINEWIDTH)
                    handles.append(h)
                    labels.append(model)

        visible_axes = [axes[i, j] for j in range(ncols) if axes[i, j].get_visible()]
        if visible_axes:
            apply_upper_row_ylim(visible_axes, row_vals, metric_key)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if not ax.get_visible():
                continue
            set_day_axis(ax, show_xlabel=(i == nrows - 1))

    metric_title = "RMSE" if metric_key == "rmse" else "Mean Error (Bias)"
    fig.suptitle(f"{metric_title} | GLOBAL", fontsize=13, y=0.992)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.3,
    )

    fig.tight_layout(rect=[0.06, 0.04, 1.0, 0.945])

    outpng = OUTDIR / f"global_upper_{metric_key}_panel_6hourly.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {outpng}")


# ============================================================
# Main
# ============================================================
def main():
    for metric_key in ["rmse", "bias"]:
        model_means = load_metric_means(metric_key)
        if not model_means:
            print(f"No data loaded for {metric_key}")
            continue

        plot_surface_panel(metric_key, model_means)
        plot_upper_panel(metric_key, model_means)

    print("\nDone.")


if __name__ == "__main__":
    main()
