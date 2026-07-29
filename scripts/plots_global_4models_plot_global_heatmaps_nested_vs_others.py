#!/usr/bin/env python3
"""
Global RMSE heatmaps showing Nested EAGLE improvement (%) versus:
  - GFS
  - AIGFS
  - ECMWF IFS

Only RMSE is plotted.

Positive values = Nested EAGLE improvement.
Negative values = Nested EAGLE degradation.

Blue = positive improvement.
Red  = negative improvement.

Improvement formula:
  100 * (reference RMSE - Nested EAGLE RMSE) / reference RMSE

Uses 6-hourly forecast data from 0 to 240 h.
Output PNG files are written into the same folder as this script.
"""

from pathlib import Path
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
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    "/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard_system/data_system/outputs"
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# Paths/settings
# ============================================================

SCORECARD_DIR = Path("/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard_system/data")
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR

METRIC = "rmse"

# Dynamic color scale percentile. Use 100 for full max/min range.
COLOR_PERCENTILE = 98

# Same fixed symmetric color range for surface and upper heatmaps.
# Units are percent improvement.
HEATMAP_LIMIT = 15.0

# Data are 6-hourly through D10.
FHR_WANTED = np.arange(0, 241, 6)

MODEL_INFO = {
    "Nested EAGLE": {
        "dir": SCORECARD_DIR / "nested_eagle_2025",
        "pattern": "{metric}.convobs.nested-global.2025-*_to_2025-*.nc",
    },
    "GFS": {
        "dir": SCORECARD_DIR / "gfs_zarr_2025",
        "pattern": "{metric}.convobs.global.2025-*_to_2025-*.nc",
    },
    "AIGFS": {
        "dir": SCORECARD_DIR / "aigfs_2025",
        "pattern": "{metric}.convobs.global.2025-*_to_2025-*.nc",
    },
    "ECMWF IFS": {
        "dir": SCORECARD_DIR / "ecmwf_ifs_2025",
        "pattern": "{metric}.convobs.global.2025-*_to_2025-*.nc",
    },
}

COMPARE_MODELS = ["GFS", "AIGFS", "ECMWF IFS"]

SURFACE_VARS = [
    ("2m_temperature", "T2M"),
    ("surface_pressure", "SP"),
    ("10m_zonal_wind", "U10"),
    ("10m_meridional_wind", "V10"),
    ("10m_wind_speed", "WS10"),
]

UPPER_VARS = [
    ("geopotential_height", "HGT"),
    ("temperature", "T"),
    ("specific_humidity", "Q"),
    ("zonal_wind", "U"),
    ("meridional_wind", "V"),
    ("wind_speed", "WS"),
]

LEVELS = [250, 500, 850]


# ============================================================
# Helpers
# ============================================================

def find_metric_files(model_name):
    info = MODEL_INFO[model_name]
    pattern = info["pattern"].format(metric=METRIC)
    return sorted(info["dir"].glob(pattern))


def load_yearly_mean(model_name):
    files = find_metric_files(model_name)

    if not files:
        raise FileNotFoundError(f"No {METRIC} files found for {model_name}")

    print(f"  {model_name:12s}: loading {len(files)} files")

    dsets = []
    for f in files:
        ds = xr.open_dataset(f, decode_timedelta=True)

        if "fhr" in ds.coords:
            have = np.asarray(ds["fhr"].values, dtype=int)
            use = [int(x) for x in FHR_WANTED if int(x) in have]
            ds = ds.sel(fhr=use)

        ds.load()
        dsets.append(ds)

    ds_all = xr.concat(
        dsets,
        dim="t0",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    mean_ds = ds_all.mean(dim="t0", skipna=True).load()

    for ds in dsets:
        ds.close()

    return mean_ds


def compute_rmse_improvement(nested_vals, ref_vals):
    """
    Positive = Nested EAGLE improvement.

    RMSE improvement:
      100 * (reference - Nested EAGLE) / reference
    """
    nested_vals = np.asarray(nested_vals, dtype=float)
    ref_vals = np.asarray(ref_vals, dtype=float)

    numerator = ref_vals - nested_vals
    denominator = ref_vals

    out = np.full_like(nested_vals, np.nan, dtype=float)
    good = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 0.0)
    )

    out[good] = 100.0 * numerator[good] / denominator[good]
    return out


def get_fhr_values(ds):
    if "fhr" not in ds.coords:
        raise ValueError("Dataset missing fhr coordinate")
    return np.asarray(ds["fhr"].values, dtype=int)


def extract_surface_row(nested_ds, ref_ds, varname):
    if varname not in nested_ds.data_vars:
        return None
    if varname not in ref_ds.data_vars:
        return None

    nested_da = nested_ds[varname]
    ref_da = ref_ds[varname]

    if "level" in nested_da.dims or "level" in ref_da.dims:
        return None

    return compute_rmse_improvement(nested_da.values, ref_da.values)


def extract_upper_row(nested_ds, ref_ds, varname, level):
    if varname not in nested_ds.data_vars:
        return None
    if varname not in ref_ds.data_vars:
        return None

    nested_da = nested_ds[varname]
    ref_da = ref_ds[varname]

    if "level" not in nested_da.dims:
        return None
    if "level" not in ref_da.dims:
        return None

    if level not in [int(x) for x in nested_da["level"].values]:
        return None
    if level not in [int(x) for x in ref_da["level"].values]:
        return None

    nested_vals = nested_da.sel(level=level).values
    ref_vals = ref_da.sel(level=level).values

    return compute_rmse_improvement(nested_vals, ref_vals)


def build_surface_matrix(nested_ds, ref_ds):
    rows = []
    labels = []

    for varname, short_name in SURFACE_VARS:
        row = extract_surface_row(nested_ds, ref_ds, varname)
        if row is None:
            print(f"    missing surface var: {varname}")
            continue

        rows.append(row)
        labels.append(short_name)

    if not rows:
        return None, None

    return np.vstack(rows), labels


def build_upper_matrix(nested_ds, ref_ds):
    rows = []
    labels = []

    for varname, short_name in UPPER_VARS:
        for lev in LEVELS:
            row = extract_upper_row(nested_ds, ref_ds, varname, lev)
            if row is None:
                continue

            rows.append(row)
            labels.append(f"{short_name} {lev}")

    if not rows:
        return None, None

    return np.vstack(rows), labels


def lead_time_ticks(fhr_vals):
    positions = []
    labels = []

    for day in range(0, 11):
        hour = day * 24
        idx = np.where(fhr_vals == hour)[0]
        if len(idx) > 0:
            positions.append(int(idx[0]))
            labels.append(f"D{day}")

    return positions, labels


def clean_model_name(name):
    return name.lower().replace(" ", "_")



def get_dynamic_color_limit(matrix):
    """
    Compute symmetric color limit from actual heatmap values.
    Uses percentile to avoid single outliers dominating the plot.
    """
    vals = np.asarray(matrix, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 10.0

    lim = np.nanpercentile(np.abs(vals), COLOR_PERCENTILE)

    if not np.isfinite(lim) or lim <= 0:
        lim = np.nanmax(np.abs(vals))

    if not np.isfinite(lim) or lim <= 0:
        lim = 10.0

    # Round up to a clean value.
    if lim <= 5:
        step = 1
    elif lim <= 10:
        step = 2
    elif lim <= 25:
        step = 5
    elif lim <= 50:
        step = 10
    else:
        step = 20

    lim = step * np.ceil(lim / step)
    return float(lim)


def add_cell_values(ax, matrix, lim):
    """
    Add percent values inside heatmap cells.
    White text on darker colors; black text on light colors.
    """
    vals = np.asarray(matrix, dtype=float)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return

    max_abs = np.nanmax(np.abs(finite))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue

            # Keep text readable.
            color = "white" if abs(v) > 0.45 * max_abs else "black"

            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )


def plot_heatmap(matrix, row_labels, fhr_vals, compare_model, panel_type, outfile):
    if matrix is None or row_labels is None:
        print(f"Skipping {outfile.name}: no data")
        return

    nrows = matrix.shape[0]

    if panel_type == "surface":
        figsize = (14.5, 4.8)
        left = 0.12
        bottom = 0.16
        top = 0.79
    else:
        figsize = (14.5, 8.8)
        left = 0.13
        bottom = 0.12
        top = 0.84

    fig, ax = plt.subplots(figsize=figsize)

    lim = get_dynamic_color_limit(matrix)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )

    add_cell_values(ax, matrix, lim)

    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels, fontsize=10)

    xticks, xlabels = lead_time_ticks(fhr_vals)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_xlabel("Lead time (days)", fontsize=11)

    panel_name = "Surface fields" if panel_type == "surface" else "Upper-air fields"

    main_title = f"Nested-EAGLE (global) vs {compare_model} | {panel_name}"
    subtitle = (
        "RMSE Improvement (%) | Positive values, blue colors, show Nested-EAGLE improvement | All models: 0.25 degree"
    )
    note = "Improvement = 100 x (reference RMSE - Nested EAGLE RMSE) / reference RMSE"

    fig.suptitle(main_title, fontsize=13, y=0.970, fontweight="bold")
    ax.set_title(subtitle, fontsize=10.5, pad=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label(f"Improvement (%) | scale: ±{lim:g}", fontsize=10)
    cbar.set_ticks(
        [
            -lim,
            -lim / 2,
            0,
            lim / 2,
            lim,
        ]
    )
    cbar.ax.tick_params(labelsize=9)

    fig.text(
        0.5,
        0.025,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.subplots_adjust(left=left, right=0.93, top=top, bottom=bottom)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {outfile}")


# ============================================================
# Main
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading yearly mean RMSE files")

    yearly = {}
    for model_name in ["Nested EAGLE"] + COMPARE_MODELS:
        yearly[model_name] = load_yearly_mean(model_name)

    fhr_vals = get_fhr_values(yearly["Nested EAGLE"])

    for compare_model in COMPARE_MODELS:
        print(f"\nPlotting Nested EAGLE vs {compare_model}")

        surface_matrix, surface_labels = build_surface_matrix(
            yearly["Nested EAGLE"],
            yearly[compare_model],
        )

        surface_outfile = OUTDIR / (
            f"rmse_surface_heatmap_nested_eagle_vs_"
            f"{clean_model_name(compare_model)}_global.png"
        )

        plot_heatmap(
            surface_matrix,
            surface_labels,
            fhr_vals,
            compare_model,
            "surface",
            surface_outfile,
        )

        upper_matrix, upper_labels = build_upper_matrix(
            yearly["Nested EAGLE"],
            yearly[compare_model],
        )

        upper_outfile = OUTDIR / (
            f"rmse_upper_heatmap_nested_eagle_vs_"
            f"{clean_model_name(compare_model)}_global.png"
        )

        plot_heatmap(
            upper_matrix,
            upper_labels,
            fhr_vals,
            compare_model,
            "upper",
            upper_outfile,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
