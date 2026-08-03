#!/usr/bin/env python3
"""
Nested-EAGLE-Global (AI) vs AIFS regional RMSE improvement heatmap using global 0.25 degree data.

Creates a region-comparison heatmap similar to the reference plot.

Regions:
  global
  northern_hemisphere
  southern_hemisphere
  conus

Lead times:
  D1-D10, using forecast hours 24, 48, ..., 240

Metric:
  RMSE improvement (%) = 100 * (AIFS_RMSE - NestedEAGLE_RMSE) / AIFS_RMSE

Positive values mean Nested-EAGLE has lower RMSE than GFS.
Blue = improvement / better.
Red  = degradation / worse.

Output:
  /scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv/scorecard/plots_nested_eagle_vs_gfs_regions/nested_eagle_global_vs_aifs_regions_rmse_improvement_D1_D10.png
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Paths
# ============================================================

BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "input_data"))
OUTDIR = Path(__import__("os").environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))

NESTED_DIR = BASE / "nested_eagle_global_2025"
GFS_DIR = BASE / "aifs_2025"

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPNG = OUTDIR / "nested_eagle_global_vs_aifs_regions_rmse_improvement_D1_D10.png"


# ============================================================
# Configuration
# ============================================================

REGIONS = [
    {
        "key": "global",
        "title": "global",
        "aliases": ["global", "GLOBAL", "all", "ALL"],
    },
    {
        "key": "northern_hemisphere",
        "title": "northern_hemisphere",
        "aliases": [
            "northern_hemisphere",
            "northern hemisphere",
            "north_hemisphere",
            "nh",
            "NH",
            "nhem",
            "NHEM",
        ],
    },
    {
        "key": "southern_hemisphere",
        "title": "southern_hemisphere",
        "aliases": [
            "southern_hemisphere",
            "southern hemisphere",
            "south_hemisphere",
            "sh",
            "SH",
            "shem",
            "SHEM",
        ],
    },
    {
        "key": "conus",
        "title": "conus",
        "aliases": ["conus", "CONUS", "us", "US"],
    },
]

DAYS = list(range(1, 11))
FHRS = [24 * d for d in DAYS]
XLABELS = [f"D{d}" for d in DAYS]

# Variable rows. For upper-air variables, "levels" gives pressure levels in hPa.
# Surface variables have levels=None.
ROW_GROUPS = [
    {
        "name": "geopotential_height",
        "rows": [
            {"var": "geopotential_height", "label": "geopotential_height 100", "levels": [100]},
            {"var": "geopotential_height", "label": "geopotential_height 250", "levels": [250]},
            {"var": "geopotential_height", "label": "geopotential_height 500", "levels": [500]},
            {"var": "geopotential_height", "label": "geopotential_height 850", "levels": [850]},
        ],
    },
    {
        "name": "zonal_wind",
        "rows": [
            {"var": "zonal_wind", "label": "zonal_wind 100", "levels": [100]},
            {"var": "zonal_wind", "label": "zonal_wind 250", "levels": [250]},
            {"var": "zonal_wind", "label": "zonal_wind 500", "levels": [500]},
            {"var": "zonal_wind", "label": "zonal_wind 850", "levels": [850]},
        ],
    },
    {
        "name": "meridional_wind",
        "rows": [
            {"var": "meridional_wind", "label": "meridional_wind 100", "levels": [100]},
            {"var": "meridional_wind", "label": "meridional_wind 250", "levels": [250]},
            {"var": "meridional_wind", "label": "meridional_wind 500", "levels": [500]},
            {"var": "meridional_wind", "label": "meridional_wind 850", "levels": [850]},
        ],
    },
    {
        "name": "temperature",
        "rows": [
            {"var": "temperature", "label": "temperature 100", "levels": [100]},
            {"var": "temperature", "label": "temperature 250", "levels": [250]},
            {"var": "temperature", "label": "temperature 500", "levels": [500]},
            {"var": "temperature", "label": "temperature 850", "levels": [850]},
        ],
    },
    {
        "name": "specific_humidity",
        "rows": [
            {"var": "specific_humidity", "label": "specific_humidity 100", "levels": [100]},
            {"var": "specific_humidity", "label": "specific_humidity 250", "levels": [250]},
            {"var": "specific_humidity", "label": "specific_humidity 500", "levels": [500]},
            {"var": "specific_humidity", "label": "specific_humidity 850", "levels": [850]},
        ],
    },
    {
        "name": "wind_speed",
        "rows": [
            {"var": "wind_speed", "label": "wind_speed 100", "levels": [100]},
            {"var": "wind_speed", "label": "wind_speed 250", "levels": [250]},
            {"var": "wind_speed", "label": "wind_speed 500", "levels": [500]},
            {"var": "wind_speed", "label": "wind_speed 850", "levels": [850]},
        ],
    },
    {
        "name": "surface",
        "rows": [
            {"var": "10m_zonal_wind", "label": "10m_zonal_wind", "levels": None},
            {"var": "10m_meridional_wind", "label": "10m_meridional_wind", "levels": None},
            {"var": "2m_temperature", "label": "2m_temperature", "levels": None},
            {"var": "10m_wind_speed", "label": "10m_wind_speed", "levels": None},
        ],
    },
]

VAR_ALIASES = {
    "geopotential_height": ["geopotential_height", "geopotential", "gh", "z"],
    "zonal_wind": ["zonal_wind", "u_wind", "u_component_of_wind", "u"],
    "meridional_wind": ["meridional_wind", "v_wind", "v_component_of_wind", "v"],
    "temperature": ["temperature", "tmp", "t"],
    "specific_humidity": ["specific_humidity", "q", "spfh"],
    "wind_speed": ["wind_speed", "wind"],
    "surface_pressure": ["surface_pressure", "pressure_surface", "sp", "ps"],
    "10m_zonal_wind": ["10m_zonal_wind", "u10", "10u"],
    "10m_meridional_wind": ["10m_meridional_wind", "v10", "10v"],
    "2m_temperature": ["2m_temperature", "t2m", "2t"],
    "2m_specific_humidity": ["2m_specific_humidity", "q2m", "2m_q"],
    "10m_wind_speed": ["10m_wind_speed", "wind_speed_10m", "ws10"],
}

REGION_DIM_CANDIDATES = [
    "region",
    "mask",
    "vx_mask",
    "mask_name",
    "stat_region",
    "domain",
    "area",
]

FHR_CANDIDATES = [
    "fhr",
    "lead",
    "lead_time",
    "forecast_hour",
    "fhour",
]

LEVEL_CANDIDATES = [
    "level",
    "lev",
    "pressure",
    "pressure_level",
    "plev",
    "isobaricInhPa",
]


# ============================================================
# File discovery
# ============================================================

def candidate_patterns(model_key, region_key):
    """
    Strict file matching for region-comparison plot.

    This avoids mixing southern_hemisphere with polar_south or tropics files.
    """

    if model_key == "nested":
        prefix_list = [
            "rmse.convobs.nested-global",
            "rmse.convobs.global",
        ]
    elif model_key == "gfs":
        prefix_list = [
            "rmse.convobs.global",
        ]
    else:
        raise ValueError(model_key)

    if region_key == "global":
        base_patterns = [
            f"{prefix}.nc"
            for prefix in prefix_list
        ]
        regional_patterns = []
        return base_patterns, regional_patterns

    region_name = region_key

    base_patterns = []
    regional_patterns = [
        f"{prefix}.{region_name}.nc"
        for prefix in prefix_list
    ]

    return base_patterns, regional_patterns

def files_for_model_region(model_key, region_key):
    directory = NESTED_DIR if model_key == "nested" else GFS_DIR
    base_patterns, regional_patterns = candidate_patterns(model_key, region_key)

    pattern_sets = []

    base_files = []
    for pat in base_patterns:
        base_files.extend(sorted(directory.glob(pat)))

    regional_files = []
    for pat in regional_patterns:
        regional_files.extend(sorted(directory.glob(pat)))

    if region_key == "global":
        if base_files:
            pattern_sets.append(("base", sorted(set(base_files))))
    else:
        if regional_files:
            pattern_sets.append(("regional", sorted(set(regional_files))))

    return pattern_sets


# ============================================================
# Dataset helpers
# ============================================================

def decode_value(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


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
    aliases = VAR_ALIASES.get(var_key, [var_key])

    for name in aliases:
        if name in ds.data_vars:
            return name

    return None


def normalize_fhr(da):
    fhr_name = find_name(list(da.coords) + list(da.dims), FHR_CANDIDATES)

    if fhr_name is None:
        return None

    if fhr_name != "fhr":
        da = da.rename({fhr_name: "fhr"})

    return da


def normalize_level(da):
    lev_name = find_name(list(da.coords) + list(da.dims), LEVEL_CANDIDATES)

    if lev_name is None:
        return da, None

    if lev_name != "level":
        da = da.rename({lev_name: "level"})

    return da, "level"


def select_region(da, region_info, allow_no_region_for_global):
    names = list(da.coords) + list(da.dims)
    region_dim = find_name(names, REGION_DIM_CANDIDATES)

    if region_dim is None:
        # If this is already a region-specific file, the filename defines the region.
        # In that case there is no internal region coordinate to select.
        if allow_no_region_for_global:
            return da
        return None

    vals = [decode_value(x).strip() for x in da[region_dim].values]
    vals_lower = [v.lower().replace("-", "_").replace(" ", "_") for v in vals]

    aliases = []
    for a in region_info["aliases"]:
        aliases.append(a.lower().replace("-", "_").replace(" ", "_"))

    match_index = None
    for i, v in enumerate(vals_lower):
        if v in aliases:
            match_index = i
            break

    if match_index is None:
        return None

    selected_value = da[region_dim].values[match_index]
    return da.sel({region_dim: selected_value})


def select_level(da, levels):
    da, lev_name = normalize_level(da)

    if levels is None:
        if lev_name is not None and "level" in da.dims:
            return None
        return da

    if lev_name is None:
        return None

    have = [int(x) for x in da["level"].values]

    for lev in levels:
        if int(lev) in have:
            return da.sel(level=int(lev))

    return None


def mean_series_from_files(model_key, region_info, row):
    pattern_sets = files_for_model_region(model_key, region_info["key"])

    if not pattern_sets:
        print(f"WARNING: no file candidates for {model_key} {region_info['key']}")
        return np.full(len(FHRS), np.nan)

    var_key = row["var"]
    levels = row["levels"]

    for source_type, files in pattern_sets:
        arrays = []
        allow_no_region = source_type == "regional" or region_info["key"] == "global"

        for f in files:
            ds = open_dataset(f)

            ds_var = find_var(ds, var_key)
            if ds_var is None:
                ds.close()
                continue

            da = ds[ds_var]

            da = normalize_fhr(da)
            if da is None or "fhr" not in da.coords:
                ds.close()
                continue

            da = select_region(da, region_info, allow_no_region_for_global=allow_no_region)
            if da is None:
                ds.close()
                continue

            da = select_level(da, levels)
            if da is None:
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

        if arrays:
            all_da = xr.concat(
                arrays,
                dim="t0_all",
                coords="minimal",
                compat="override",
                join="outer",
                combine_attrs="drop_conflicts",
            )

            mean_dims = [d for d in all_da.dims if d != "fhr"]
            mean = all_da.mean(mean_dims, skipna=True)
            mean = mean.reindex(fhr=FHRS)

            vals = np.asarray(mean.values, dtype=float).squeeze()

            if vals.shape != (len(FHRS),):
                print(
                    f"WARNING: bad shape for {model_key} {region_info['key']} "
                    f"{row['label']}: {vals.shape}"
                )
                return np.full(len(FHRS), np.nan)

            return vals

    print(f"WARNING: no usable data for {model_key} {region_info['key']} {row['label']}")
    return np.full(len(FHRS), np.nan)


def build_rows():
    rows = []
    group_breaks = []

    for group in ROW_GROUPS:
        for row in group["rows"]:
            # Skip 100 hPa rows for cleaner scorecard display.
            if row.get("levels") == [100]:
                continue
            rows.append(row)
        group_breaks.append(len(rows) - 0.5)

    return rows, group_breaks[:-1]


def build_region_matrix(region_info, rows):
    matrix = np.full((len(rows), len(FHRS)), np.nan)

    for i, row in enumerate(rows):
        print(f"Region={region_info['key']:20s} row={row['label']}")

        nested = mean_series_from_files("nested", region_info, row)
        gfs = mean_series_from_files("gfs", region_info, row)

        with np.errstate(invalid="ignore", divide="ignore"):
            improvement = 100.0 * (gfs - nested) / gfs

        improvement[~np.isfinite(improvement)] = np.nan
        matrix[i, :] = improvement

    return matrix


def choose_box_text_color(value, vmax):
    """Choose black/white text for readability inside heatmap cells."""
    if not np.isfinite(value):
        return "black"
    if vmax <= 0:
        return "black"

    # White text on strong blue/red cells, black near white/neutral cells.
    if abs(value) >= 0.60 * vmax:
        return "white"
    return "black"


# ============================================================
# Plot
# ============================================================

def plot_heatmap(region_mats, rows, group_breaks):
    nrows = len(rows)
    ncols = len(FHRS)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(REGIONS),
        figsize=(21, 10.5),
        sharey=True,
        gridspec_kw={
            "left": 0.105,
            "right": 0.925,
            "top": 0.86,
            "bottom": 0.12,
            "wspace": 0.055,
        },
    )

    fig.suptitle(
        "Nested-EAGLE-Global (AI) vs AIFS | Global 0.25 degree data",
        fontsize=19,
        fontweight="bold",
        y=0.975,
    )

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("#f2f2f2")

    vmin = -30.0
    vmax = 30.0

    for ax, region_info in zip(axes, REGIONS):
        mat = region_mats[region_info["key"]]

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(region_info["title"], fontsize=13, fontweight="bold", pad=12)

        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)

        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(XLABELS, fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)

        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in group_breaks:
            ax.axhline(y, color="#333333", linewidth=1.15)

        # Add one-decimal RMSE improvement values in each cell.
        for ii in range(nrows):
            for jj in range(ncols):
                value = mat[ii, jj]
                if not np.isfinite(value):
                    continue
                ax.text(
                    jj,
                    ii,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color=choose_box_text_color(value, vmax),
                    clip_on=True,
                )

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.9)

    ylabels = [r["label"] for r in rows]
    axes[0].set_yticks(np.arange(nrows))
    axes[0].set_yticklabels(ylabels, fontsize=10)
    axes[0].tick_params(axis="y", length=0, pad=6)

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)

    cax = fig.add_axes([0.945, 0.18, 0.022, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("RMSE improvement (%)", fontsize=12, rotation=90, labelpad=16)
    cb.set_ticks([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
    cb.ax.tick_params(labelsize=10)

    fig.text(
        0.5,
        0.055,
        "RMSE improvement (%) = 100 x (AIFS RMSE - Nested-EAGLE-Global (AI) RMSE) / AIFS RMSE. Positive/blue values indicate lower RMSE for Nested-EAGLE-Global (AI). Cell values are RMSE improvement percentages.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    fig.savefig(OUTPNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUTPNG}")


def main():
    rows, group_breaks = build_rows()

    region_mats = {}
    for region_info in REGIONS:
        print("")
        print("=" * 90)
        print(f"Building region panel: {region_info['key']}")
        print("=" * 90)
        region_mats[region_info["key"]] = build_region_matrix(region_info, rows)

    plot_heatmap(region_mats, rows, group_breaks)


if __name__ == "__main__":
    main()
