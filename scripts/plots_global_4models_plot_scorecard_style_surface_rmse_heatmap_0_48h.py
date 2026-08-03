#!/usr/bin/env python3
"""
Scorecard-style surface RMSE heatmap.

Variables:
  1. 2m Temperature RMSE
  2. Surface Pressure RMSE
  3. 10m Wind Speed RMSE

Lead times:
  0, 12, 24, 36, 48 h

Rows:
  AI Global:
    - Nested-EAGLE (Global)
    - AIGFS
    - AIFS

  AI-HR:
    - Nested-EAGLE-LAM

  Physical:
    - GFS (Global)
    - GFS-CONUS
    - HRRR

Output:
  scorecard_style_surface_rmse_0_48h.png
"""

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

import matplotlib as mpl

# ----------------------------------------------------------------------
# scorecard_system path control
# Input data are read from scorecard_system/data.
# All generated products are written to scorecard_system/outputs.
# ----------------------------------------------------------------------
import os as _scorecard_os
from pathlib import Path as _scorecard_Path
SCORECARD_SYSTEM_DATA_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "input_data")
))
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "outputs")
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# Paths
# ============================================================

BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "input_data"))
OUTDIR = SCORECARD_SYSTEM_OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPNG = OUTDIR / "scorecard_style_surface_rmse_0_48h.png"


# ============================================================
# Configuration
# ============================================================

FHRS = [0, 12, 24, 36, 48]

VAR_PANELS = [
    {
        "key": "2m_temperature",
        "title": "2m Temperature",
        "subtitle": "2m temperature RMSE [K]",
        "scale": 1.0,
        "fmt": "{:.2f}",
    },
    {
        "key": "surface_pressure",
        "title": "Surface Pressure",
        "subtitle": "surface pressure RMSE [hPa]",
        "scale": "auto_hpa",
        "fmt": "{:.2f}",
    },
    {
        "key": "10m_wind_speed",
        "title": "Wind Speed 10m",
        "subtitle": "10m wind speed RMSE [m/s]",
        "scale": 1.0,
        "fmt": "{:.2f}",
    },
]

MODEL_GROUPS = [
    {
        "label": "AI\nGlobal\nModel",
        "models": [
            "Nested-EAGLE (Global)",
            "AIGFS",
            "AIFS",
        ],
        "bar_color": "#dbeafe",
        "bar_text_color": "#1f4e79",
    },
    {
        "label": "AI-HR\nModel",
        "models": [
            "Nested-EAGLE-LAM",
        ],
        "bar_color": "#ede9fe",
        "bar_text_color": "#5b21b6",
    },
    {
        "label": "Physical\nModel",
        "models": [
            "GFS (Global)",
            "GFS-CONUS",
            "HRRR",
        ],
        "bar_color": "#e2f0d9",
        "bar_text_color": "#3f6b45",
    },
]

MODEL_INFO = {
    "Nested-EAGLE (Global)": {
        "dir": BASE / "nested_eagle_global_2025",
        "pattern": "rmse.convobs.nested-global.nc",
    },
    "AIGFS": {
        "dir": BASE / "aigfs_2025",
        "pattern": "rmse.convobs.global.nc",
    },
    "AIFS": {
        "dir": BASE / "aifs_2025",
        "pattern": "rmse.convobs.global.nc",
    },
    "Nested-EAGLE-LAM": {
        "dir": BASE / "nested_eagle_lam_2025",
        "pattern": "rmse.convobs.nested-lam.nc",
    },
    "GFS (Global)": {
        "dir": BASE / "gfs_2025",
        "pattern": "rmse.convobs.global.nc",
    },
    "GFS-CONUS": {
        "dir": BASE / "gfs_2025",
        "pattern": "rmse.convobs.global.conus.nc",
    },
    "HRRR": {
        "dir": BASE / "hrrr_2025",
        "pattern": "rmse.convobs.lam.nc",
    },
}


# ============================================================
# Data helpers
# ============================================================

def all_models():
    names = []
    for group in MODEL_GROUPS:
        names.extend(group["models"])
    return names


def open_dataset(path):
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)


def apply_scale(vals, scale):
    vals = np.asarray(vals, dtype=float)

    if scale == "auto_hpa":
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmedian(np.abs(finite)) > 20:
            return vals / 100.0
        return vals

    return vals * float(scale)


def load_model_var_series(model_name, panel):
    """
    Return annual mean RMSE values for one model and one surface variable.
    Output shape must be: (len(FHRS),)
    """
    info = MODEL_INFO[model_name]
    files = sorted(info["dir"].glob(info["pattern"]))

    if not files:
        print(f"WARNING: no files for {model_name}: {info['dir']}/{info['pattern']}")
        return np.full(len(FHRS), np.nan)

    varname = panel["key"]

    if skip_aigfs_aifs_derived_surface(model_name, varname):
        print(f"Skipping derived surface field for {model_name}: {varname}")
        return np.full(len(FHRS), np.nan)

    arrays = []

    for f in files:
        ds = open_dataset(f)

        if varname not in ds.data_vars:
            ds.close()
            continue

        da = ds[varname]

        if "fhr" not in da.coords:
            ds.close()
            continue

        # Surface variables should not have level dimension.
        if "level" in da.dims:
            ds.close()
            continue

        have_fhr = [int(x) for x in da["fhr"].values]
        use_fhr = [int(x) for x in FHRS if int(x) in have_fhr]

        if not use_fhr:
            ds.close()
            continue

        da = da.sel(fhr=use_fhr)
        da = da.reindex(fhr=FHRS)

        if "t0" not in da.dims:
            da = da.expand_dims("t0")

        arrays.append(da.load())
        ds.close()

    if not arrays:
        print(f"WARNING: no usable data for {model_name} {varname}")
        return np.full(len(FHRS), np.nan)

    all_da = xr.concat(
        arrays,
        dim="t0_all",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="drop_conflicts",
    )

    # Average over every dimension except forecast hour.
    # This handles monthly files and any original t0 dimension.
    mean_dims = [d for d in all_da.dims if d != "fhr"]
    mean = all_da.mean(mean_dims, skipna=True)
    mean = mean.reindex(fhr=FHRS)

    vals = np.asarray(mean.values, dtype=float).squeeze()

    if vals.shape != (len(FHRS),):
        raise ValueError(
            f"Expected shape {(len(FHRS),)} for {model_name} {varname}, "
            f"got {vals.shape}. dims={mean.dims}"
        )

    vals = apply_scale(vals, panel.get("scale", 1.0))
    return vals


def build_table():
    models = all_models()
    data = np.full((len(models), len(VAR_PANELS), len(FHRS)), np.nan)

    for i, model in enumerate(models):
        for j, panel in enumerate(VAR_PANELS):
            print(f"Loading {model:22s} | {panel['title']}")
            data[i, j, :] = load_model_var_series(model, panel)

    return models, data


# ============================================================
# Plot helpers
# ============================================================

def choose_text_color(value, vmin, vmax):
    if not np.isfinite(value):
        return "black"
    if vmax <= vmin:
        return "black"

    normed = (value - vmin) / (vmax - vmin)
    return "white" if normed < 0.18 or normed > 0.82 else "black"


def clean_label(value, fmt):
    if not np.isfinite(value):
        return ""

    label = fmt.format(value)

    if label in ("-0.00", "-0.000", "-0"):
        label = "0"

    return label


# ============================================================
# Plot
# ============================================================

def plot_scorecard(models, data):
    n_models = len(models)
    n_panels = len(VAR_PANELS)
    n_fhrs = len(FHRS)

    fig = plt.figure(figsize=(15.5, 9.8))

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2 + n_panels,
        height_ratios=[0.9, 0.55, n_models],
        width_ratios=[1.35, 2.15] + [2.75] * n_panels,
        hspace=0.02,
        wspace=0.040,
    )

    fig.suptitle(
        "Surface RMSE Scorecard: AI and Physical Model Comparison",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )

    fig.text(
        0.5,
        0.925,
        "Surface fields | 0-48 h lead time | Blue = lower RMSE / better",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color="#4b5563",
    )

    # Variable headers
    for j, panel in enumerate(VAR_PANELS):
        axh = fig.add_subplot(gs[0, 2 + j])
        axh.axis("off")

        axh.text(
            0.5,
            0.52,
            panel["title"],
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

        axh.text(
            0.5,
            0.13,
            panel["subtitle"],
            ha="center",
            va="center",
            fontsize=10,
        )

    # Lead-time header
    for j in range(n_panels):
        axlead = fig.add_subplot(gs[1, 2 + j])
        axlead.set_xlim(-0.5, n_fhrs - 0.5)
        axlead.set_ylim(0, 1)
        axlead.axis("off")

        rect = mpl.patches.Rectangle(
            (-0.5, 0.02),
            n_fhrs,
            0.90,
            facecolor="#f8fafc",
            edgecolor="#b7b7b7",
            linewidth=0.8,
        )
        axlead.add_patch(rect)

        for k, fhr in enumerate(FHRS):
            axlead.text(
                k,
                0.50,
                f"{fhr}h",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    # Left group labels
    ax_group = fig.add_subplot(gs[2, 0])
    ax_group.set_xlim(0, 1)
    ax_group.set_ylim(n_models - 0.5, -0.5)
    ax_group.axis("off")

    row_start = 0
    sep_lines = []

    for group in MODEL_GROUPS:
        n = len(group["models"])
        y0 = row_start - 0.5

        rect = mpl.patches.Rectangle(
            (0, y0),
            1,
            n,
            facecolor=group["bar_color"],
            edgecolor="#b7b7b7",
            linewidth=0.8,
        )
        ax_group.add_patch(rect)

        ax_group.text(
            0.5,
            row_start + (n - 1) / 2,
            group["label"],
            ha="center",
            va="center",
            fontsize=15.0,
            fontweight="bold",
            rotation=90,
            color=group["bar_text_color"],
            clip_on=True,
            linespacing=0.85,
            wrap=True,
        )

        row_start += n
        sep_lines.append(row_start - 0.5)

    # Model labels
    ax_models = fig.add_subplot(gs[2, 1])
    ax_models.set_xlim(0, 1)
    ax_models.set_ylim(n_models - 0.5, -0.5)
    ax_models.axis("off")

    for i, model in enumerate(models):
        ax_models.text(
            0.95,
            i,
            model,
            ha="right",
            va="center",
            fontsize=10.6,
            fontweight="bold",
        )

    for y in sep_lines[:-1]:
        ax_models.axhline(y, color="#b7b7b7", linewidth=1.0)

    # Heatmaps
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("#eeeeee")

    for j, panel in enumerate(VAR_PANELS):
        ax = fig.add_subplot(gs[2, 2 + j])
        mat = data[:, j, :]

        finite = mat[np.isfinite(mat)]
        if finite.size:
            vmin = np.nanmin(finite)
            vmax = np.nanmax(finite)
        else:
            vmin, vmax = 0.0, 1.0

        ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xlim(-0.5, n_fhrs - 0.5)
        ax.set_ylim(n_models - 0.5, -0.5)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticks(np.arange(-0.5, n_fhrs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in sep_lines[:-1]:
            ax.axhline(y, color="#b7b7b7", linewidth=1.0)

        fmt = panel.get("fmt", "{:.2f}")

        for i in range(n_models):
            for k in range(n_fhrs):
                v = mat[i, k]
                label = clean_label(v, fmt)
                color = choose_text_color(v, vmin, vmax)

                ax.text(
                    k,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=color,
                    clip_on=True,
                )

        for spine in ax.spines.values():
            spine.set_edgecolor("#b7b7b7")
            spine.set_linewidth(0.8)

    # Color legend moved closer to figure, above note
    cax = fig.add_axes([0.32, 0.095, 0.36, 0.024])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap=cmap)
    cax.set_xticks([0, 128, 255])
    cax.set_xticklabels(
        ["Lower RMSE\n(Better)", "Mid", "Higher RMSE\n(Worse)"],
        fontsize=10.5,
    )
    cax.set_yticks([])

    for spine in cax.spines.values():
        spine.set_visible(False)

    # Footer note below color legend
    fig.text(
        0.5,
        0.035,
        "Box values show annual mean RMSE versus conventional observations for each model, surface variable, and lead time. Lower RMSE is better. Each variable block uses its own color scale because RMSE units differ.",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#374151",
    )

    plt.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.885,
        bottom=0.165,
    )

    fig.savefig(OUTPNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUTPNG}")


def main():
    models, data = build_table()
    plot_scorecard(models, data)


if __name__ == "__main__":
    main()
