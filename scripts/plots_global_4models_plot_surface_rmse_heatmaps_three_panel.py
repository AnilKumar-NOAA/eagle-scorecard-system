#!/usr/bin/env python3
"""
Create one 3-panel surface RMSE improvement heatmap figure.

Comparisons:
  1) Nested-EAGLE vs GFS
  2) Nested-EAGLE vs AIGFS
  3) Nested-EAGLE vs AIFS

Each heatmap panel is kept close to 5 inch x 5 inch.

Improvement (%) = 100 * (baseline_rmse - nested_rmse) / baseline_rmse

Positive values (blue)  -> Nested-EAGLE improves over baseline
Negative values (red)   -> Nested-EAGLE is worse than baseline

Output:
  surface_rmse_improvement_heatmaps_3panel.png
"""

from pathlib import Path
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# scorecard_system output control
# All figures, CSVs, and text products from this script are written here.
# Input data remain under the shared scorecard data directory.
# ----------------------------------------------------------------------
import os as _scorecard_os
from pathlib import Path as _scorecard_Path
SCORECARD_SYSTEM_DATA_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data/new_data")
))
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "outputs")
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Paths
# ============================================================

BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "data/new_data"))
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# User settings
# ============================================================

#HERE = Path(__file__).resolve().parent
#SCORECARD_DIR = HERE.parent

MODEL_DIRS = {
    "Nested-EAGLE": BASE / "nested_eagle_global_2025",
    "GFS": BASE / "gfs_2025",
    "AIGFS": BASE / "aigfs_2025",
    "AIFS": BASE / "aifs_2025",
}

SURFACE_VARS = [
    "2m_temperature",
    "10m_zonal_wind",
    "10m_meridional_wind",
    "10m_wind_speed",
]

VAR_LABELS = {
    "2m_temperature": "T2M",
    "surface_pressure": "SP",
    "10m_zonal_wind": "U10",
    "10m_meridional_wind": "V10",
    "10m_wind_speed": "WSPD10",
    "2m_specific_humidity": "Q2M",
}

BASELINES = ["GFS", "AIGFS", "AIFS"]

METRIC = "rmse"

# Fixed color scale for all three panels
VMIN = -30.0
VMAX = 30.0

OUTPUT_FIG = OUTDIR / "surface_rmse_improvement_heatmaps_3panel.png"

DATE_RE = r"2025-\d{2}-\d{2}_to_2025-\d{2}-\d{2}"


# ============================================================
# Data helpers
# ============================================================

def find_monthly_metric_files(model_name, metric="rmse"):
    """
    Return only global monthly files.

    Nested-EAGLE:
      rmse.convobs.nested-global...nc

    Other global models:
      rmse.convobs.global...nc

    This excludes regional files such as .conus, .europe, etc.
    """
    model_dir = MODEL_DIRS[model_name]

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    files = []

    for p in sorted(model_dir.glob(f"{metric}.convobs*.nc")):
        name = p.name

        if model_name == "Nested-EAGLE":
            pat = rf"^{metric}\.convobs\.nested-global\.nc$"
        else:
            pat = rf"^{metric}\.convobs\.global\.nc$"

        if re.match(pat, name):
            files.append(p)

    if not files:
        raise FileNotFoundError(
            f"No global monthly {metric} files found for {model_name} in {model_dir}"
        )

    return files


def load_annual_mean_surface_series(model_name, var_name, metric="rmse"):
    """
    Load all monthly files for one model and one surface variable.
    Concatenate all t0 values across months and compute annual mean as a function of fhr.
    Returns DataArray with dimension fhr.
    """
    files = find_monthly_metric_files(model_name, metric=metric)

    pieces = []

    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4", decode_timedelta=True)

        if var_name not in ds.data_vars:
            ds.close()
            continue

        da = ds[var_name]

        # Surface variables should not have level dimension.
        if "level" in da.dims:
            ds.close()
            continue

        if "t0" not in da.dims:
            da = da.expand_dims(dim={"t0": [0]})

        pieces.append(da.load())
        ds.close()

    if not pieces:
        return None

    da_all = xr.concat(pieces, dim="t0_all")
    da_mean = da_all.mean(dim="t0_all", skipna=True)

    if "fhr" not in da_mean.dims:
        return None

    return da_mean


def compute_improvement_percent(nested_da, baseline_da):
    """
    Improvement (%) = 100 * (baseline - nested) / baseline

    Positive = Nested-EAGLE lower RMSE than baseline.
    Negative = Nested-EAGLE higher RMSE than baseline.
    """
    baseline_aligned, nested_aligned = xr.align(baseline_da, nested_da, join="inner")

    out = 100.0 * (baseline_aligned - nested_aligned) / baseline_aligned
    out = out.where(np.isfinite(out))

    return out


def build_heatmap_matrix(nested_name, baseline_name, variables):
    """
    Build matrix [n_surface_variables, n_forecast_hours] of RMSE improvement values.
    """
    row_arrays = []
    row_labels = []
    target_fhr = None

    for var in variables:
        nested_da = load_annual_mean_surface_series(nested_name, var, metric=METRIC)
        base_da = load_annual_mean_surface_series(baseline_name, var, metric=METRIC)

        label = VAR_LABELS.get(var, var)

        if nested_da is None or base_da is None:
            row_arrays.append(None)
            row_labels.append(label)
            continue

        imp = compute_improvement_percent(nested_da, base_da)

        if target_fhr is None:
            target_fhr = imp["fhr"].values

        imp = imp.reindex(fhr=target_fhr)

        row_arrays.append(imp.values.astype(float))
        row_labels.append(label)

    if target_fhr is None:
        raise RuntimeError(f"Could not determine forecast hours for {baseline_name}")

    filled_rows = []

    for arr in row_arrays:
        if arr is None:
            filled_rows.append(np.full(len(target_fhr), np.nan, dtype=float))
        else:
            filled_rows.append(arr)

    data = np.vstack(filled_rows)

    return data, np.asarray(target_fhr), row_labels


# ============================================================
# Plot helpers
# ============================================================

def get_day_ticks(fhr_vals):
    tick_hours = np.arange(0, 241, 24)
    tick_pos = []
    tick_lab = []

    for th in tick_hours:
        idx = np.where(fhr_vals == th)[0]
        if len(idx) > 0:
            tick_pos.append(int(idx[0]))
            tick_lab.append(f"D{th // 24}" if th > 0 else "D0")

    return tick_pos, tick_lab


def style_heatmap_axis(ax, fhr_vals, ylabels, title, show_ylabels=False):
    ax.set_title(title, fontsize=12, pad=8)

    tick_pos, tick_lab = get_day_ticks(fhr_vals)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=9)

    ax.set_yticks(np.arange(len(ylabels)))

    if show_ylabels:
        ax.set_yticklabels(ylabels, fontsize=10)
    else:
        ax.set_yticklabels([])

    # Cell grid
    ax.set_xticks(np.arange(-0.5, len(fhr_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================
# Main
# ============================================================

def main():
    print("Building surface RMSE improvement heatmaps ...")

    heatmaps = []
    titles = []

    fhr_vals = None
    ylabels = None

    for baseline in BASELINES:
        print(f"  comparing Nested-EAGLE vs {baseline}")

        data, fhr_vals_this, ylabels_this = build_heatmap_matrix(
            "Nested-EAGLE",
            baseline,
            SURFACE_VARS,
        )

        heatmaps.append(data)
        titles.append(f"Nested-EAGLE vs {baseline}")

        if fhr_vals is None:
            fhr_vals = fhr_vals_this
        if ylabels is None:
            ylabels = ylabels_this

    # ------------------------------------------------------------
    # Figure layout:
    #   top-left     : GFS
    #   top-right    : AIGFS
    #   bottom-center: AIFS
    #
    # Each heatmap is kept close to 5 x 5 inches.
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(11.0, 11.0))

    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.38,
        wspace=0.28,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis("off")

    axes = [ax1, ax2, ax3]

    # Shift bottom panel toward center so it does not sit too far left.
    # These values can be tuned if needed.
    pos = ax3.get_position()
    ax3.set_position([0.30, pos.y0, 0.40, pos.height])

    cmap = plt.cm.bwr_r.copy()
    cmap.set_bad(color="#d9d9d9")

    im = None

    for i, ax in enumerate(axes):
        im = ax.imshow(
            heatmaps[i],
            aspect="auto",
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
            interpolation="nearest",
        )

        # Force each panel to be square-like.
        ax.set_box_aspect(1)

        style_heatmap_axis(
            ax,
            fhr_vals=fhr_vals,
            ylabels=ylabels,
            title=titles[i],
            show_ylabels=(i in [0, 2]),
        )

    # Main title and explanation
    fig.suptitle(
        "Surface RMSE Improvement Heatmaps",
        fontsize=15,
        fontweight="bold",
        y=0.970,
    )

    fig.text(
        0.5,
        0.940,
        "Global verification | 6-hourly lead times | Positive values (blue) indicate Nested-EAGLE improvement; red indicates degradation",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.065,
        "Lead time (days)",
        ha="center",
        fontsize=11,
    )

    fig.text(
        0.045,
        0.52,
        "Surface variable",
        va="center",
        rotation="vertical",
        fontsize=11,
    )

    # Common colorbar for all three panels
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.030,
        pad=0.035,
    )

    cbar.set_label("RMSE improvement (%)", fontsize=11)
    cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
    cbar.ax.tick_params(labelsize=9)

    # Keep margins clean after manually centering the bottom panel.
    plt.subplots_adjust(
        left=0.10,
        right=0.88,
        top=0.88,
        bottom=0.12,
    )

    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
