#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import yaml
except ImportError:
    raise SystemExit("ERROR: PyYAML is required. Try: conda activate eagle")


FHR_CANDIDATES = ["fhr", "lead", "lead_time", "forecast_hour", "fhour"]
LEVEL_CANDIDATES = ["level", "lev", "pressure", "pressure_level", "plev", "isobaricInhPa"]

DERIVED_SURFACE_SKIP_VARS = {"surface_pressure", "2m_specific_humidity"}
DERIVED_SURFACE_SKIP_MODEL_KEYS = {"aigfs", "aifs"}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def find_var(ds, var_key, aliases):
    for name in aliases.get(var_key, [var_key]):
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


def build_rows(var_cfg):
    skip_levels = set(int(x) for x in var_cfg.get("skip_pressure_levels", []))

    rows = []
    group_breaks = []

    for group in var_cfg["row_groups"]:
        for row in group["rows"]:
            levels = row.get("levels")
            if levels is not None and len(levels) == 1 and int(levels[0]) in skip_levels:
                continue
            rows.append(row)
        group_breaks.append(len(rows) - 0.5)

    return rows, group_breaks[:-1]


def model_files(model_cfg, model_key, region=None, comparison_type="regional"):
    model = model_cfg["models"][model_key]
    data_base = Path(os.environ.get("SCORECARD_SYSTEM_DATA_DIR", model_cfg["data_base"]))
    directory = data_base / model["directory"]

    patterns = []

    if comparison_type == "single_domain":
        patterns.extend(model.get("default_patterns", []))
        patterns.extend(model.get("regional_patterns", []))
    else:
        if region == "global":
            patterns.extend(model.get("global_patterns", []))
        else:
            for pat in model.get("regional_patterns", []):
                patterns.append(pat.format(region=region))

    files = []
    for pat in patterns:
        files.extend(sorted(directory.glob(pat)))

    return sorted(set(files))


def mean_series(model_cfg, var_cfg, model_key, row, lead_hours, region=None, comparison_type="regional"):
    files = model_files(model_cfg, model_key, region=region, comparison_type=comparison_type)
    if not files:
        print(f"WARNING: no files for model={model_key} region={region}")
        return np.full(len(lead_hours), np.nan)

    aliases = var_cfg["variable_aliases"]
    var_key = row["var"]
    levels = row.get("levels")
    arrays = []

    for f in files:
        ds = open_dataset(f)

        ds_var = find_var(ds, var_key, aliases)
        if ds_var is None:
            ds.close()
            continue

        da = ds[ds_var]
        da = normalize_fhr(da)

        if da is None or "fhr" not in da.coords:
            ds.close()
            continue

        da = select_level(da, levels)
        if da is None:
            ds.close()
            continue

        have_fhr = [int(x) for x in da["fhr"].values]
        use_fhr = [int(x) for x in lead_hours if int(x) in have_fhr]

        if not use_fhr:
            ds.close()
            continue

        da = da.sel(fhr=use_fhr)
        da = da.reindex(fhr=lead_hours)

        if "t0" not in da.dims:
            da = da.expand_dims("t0")

        arrays.append(da.load())
        ds.close()

    if not arrays:
        print(f"WARNING: no usable data for model={model_key} region={region} row={row['label']}")
        return np.full(len(lead_hours), np.nan)

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
    mean = mean.reindex(fhr=lead_hours)

    vals = np.asarray(mean.values, dtype=float).squeeze()

    if vals.shape != (len(lead_hours),):
        print(f"WARNING: bad shape model={model_key} row={row['label']} shape={vals.shape}")
        return np.full(len(lead_hours), np.nan)

    return vals


def improvement_percent(reference, target):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100.0 * (reference - target) / reference
    out[~np.isfinite(out)] = np.nan
    return out


def variable_group(label):
    if label.startswith("geopotential_height"):
        return "geopotential_height"
    if label.startswith("zonal_wind"):
        return "zonal_wind"
    if label.startswith("meridional_wind"):
        return "meridional_wind"
    if label.startswith("temperature"):
        return "temperature"
    if label.startswith("specific_humidity"):
        return "specific_humidity"
    if label.startswith("wind_speed"):
        return "wind_speed"
    return "surface"


def recompute_group_breaks_from_rows(rows):
    breaks = []
    if not rows:
        return breaks

    last_group = variable_group(rows[0]["label"])
    for i, row in enumerate(rows[1:], start=1):
        group = variable_group(row["label"])
        if group != last_group:
            breaks.append(i - 0.5)
            last_group = group

    return breaks


def filter_rows_for_derived_surface(rows, comparison):
    target = comparison.get("target_model", "")
    reference = comparison.get("reference_model", "")

    if target not in DERIVED_SURFACE_SKIP_MODEL_KEYS and reference not in DERIVED_SURFACE_SKIP_MODEL_KEYS:
        return rows, recompute_group_breaks_from_rows(rows)

    kept = [row for row in rows if row.get("var") not in DERIVED_SURFACE_SKIP_VARS]
    return kept, recompute_group_breaks_from_rows(kept)


def choose_box_text_color(value, vmax):
    if not np.isfinite(value):
        return "black"
    if abs(value) >= 0.60 * vmax:
        return "white"
    return "black"


def write_values_csv(path, records):
    fields = [
        "comparison",
        "target_model",
        "reference_model",
        "domain",
        "resolution",
        "variable_group",
        "variable",
        "row_label",
        "level_hpa",
        "lead_hour",
        "lead_label",
        "rmse_improvement_pct",
    ]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)


def build_matrix(model_cfg, var_cfg, comparison, rows, domain):
    target = comparison["target_model"]
    ref = comparison["reference_model"]
    lead_hours = comparison["lead_hours"]

    matrix = np.full((len(rows), len(lead_hours)), np.nan)

    for i, row in enumerate(rows):
        print(f"Reading {comparison['name']} | {domain} | {row['label']}")

        target_vals = mean_series(
            model_cfg,
            var_cfg,
            target,
            row,
            lead_hours,
            region=domain,
            comparison_type=comparison["type"],
        )

        ref_vals = mean_series(
            model_cfg,
            var_cfg,
            ref,
            row,
            lead_hours,
            region=domain,
            comparison_type=comparison["type"],
        )

        matrix[i, :] = improvement_percent(ref_vals, target_vals)

    return matrix


def matrix_to_records(model_cfg, comparison, domain, resolution, rows, matrix):
    target_label = model_cfg["models"][comparison["target_model"]]["label"]
    ref_label = model_cfg["models"][comparison["reference_model"]]["label"]

    records = []

    for i, row in enumerate(rows):
        levels = row.get("levels")
        level = ""
        if levels is not None and len(levels) == 1:
            level = str(levels[0])

        for j, fhr in enumerate(comparison["lead_hours"]):
            v = matrix[i, j]
            records.append({
                "comparison": comparison["name"],
                "target_model": target_label,
                "reference_model": ref_label,
                "domain": domain,
                "resolution": resolution,
                "variable_group": variable_group(row["label"]),
                "variable": row["var"],
                "row_label": row["label"],
                "level_hpa": level,
                "lead_hour": int(fhr),
                "lead_label": comparison["lead_labels"][j],
                "rmse_improvement_pct": "" if not np.isfinite(v) else round(float(v), 3),
            })

    return records


def plot_regional(comparison, rows, group_breaks, matrices, outpng):
    regions = comparison["regions"]
    nrows = len(rows)
    ncols = len(comparison["lead_hours"])

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(regions),
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

    if len(regions) == 1:
        axes = [axes]

    fig.suptitle(comparison["title"], fontsize=18, fontweight="bold", y=0.975)

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("#f2f2f2")

    vmax = float(comparison.get("color_limit", 30))
    print(f"DEBUG: {comparison['name']} color_limit={vmax}")
    vmin = -vmax

    im = None

    for ax, region in zip(axes, regions):
        mat = matrices[region]

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(region, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)

        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(comparison["lead_labels"], fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)

        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in group_breaks:
            ax.axhline(y, color="#333333", linewidth=1.15)

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

    axes[0].set_yticks(np.arange(nrows))
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=10)
    axes[0].tick_params(axis="y", length=0, pad=6)

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)

    cax = fig.add_axes([0.945, 0.18, 0.022, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("RMSE improvement (%)", fontsize=12, rotation=90, labelpad=16)
    cb.set_ticks([-vmax, -20, -10, 0, 10, 20, vmax])
    print("DEBUG colorbar ticks:", [-vmax, -20, -10, 0, 10, 20, vmax])
    cb.ax.tick_params(labelsize=10)

    fig.text(
        0.5,
        0.055,
        "RMSE improvement (%) = 100 x (reference RMSE - target RMSE) / reference RMSE. Positive/blue values indicate lower RMSE for the target model. Cell values are percentages.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_domain(comparison, rows, group_breaks, matrix, outpng):
    nrows = len(rows)
    ncols = len(comparison["lead_hours"])

    fig, ax = plt.subplots(figsize=(13.5, 13.0))

    fig.suptitle(comparison["title"], fontsize=19, fontweight="bold", y=0.982)

    fig.text(
        0.5,
        0.952,
        comparison.get("subtitle", ""),
        ha="center",
        va="center",
        fontsize=13,
        style="italic",
        color="#4b5563",
    )

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("#f2f2f2")

    vmax = float(comparison.get("color_limit", 25))
    vmin = -vmax

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(comparison["lead_labels"], fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=6)

    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=11)
    ax.tick_params(axis="y", length=0, pad=6)

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y in group_breaks:
        ax.axhline(y, color="#222222", linewidth=1.15)

    for ii in range(nrows):
        for jj in range(ncols):
            value = matrix[ii, jj]
            if not np.isfinite(value):
                continue
            ax.text(
                jj,
                ii,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=choose_box_text_color(value, vmax),
                clip_on=True,
            )

    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
        spine.set_linewidth(0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("RMSE improvement (%)", fontsize=13, rotation=90, labelpad=16)
    cbar.ax.tick_params(labelsize=11)

    fig.text(
        0.5,
        0.035,
        "RMSE improvement (%) = 100 x (reference RMSE - target RMSE) / reference RMSE. Positive/blue values indicate lower RMSE for the target model. Cell values are percentages.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    plt.subplots_adjust(left=0.22, right=0.88, top=0.91, bottom=0.075)

    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_records(records):
    vals = []
    for r in records:
        v = r["rmse_improvement_pct"]
        if v == "":
            continue
        vals.append(float(v))

    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return "No valid values."

    return (
        f"n={vals.size}, mean={np.nanmean(vals):.2f}%, "
        f"median={np.nanmedian(vals):.2f}%, "
        f"min={np.nanmin(vals):.2f}%, max={np.nanmax(vals):.2f}%, "
        f"positive={100.0 * np.sum(vals > 0) / vals.size:.1f}%"
    )


def run_comparison(model_cfg, var_cfg, comparison, outdir):
    rows, group_breaks = build_rows(var_cfg)
    rows, group_breaks = filter_rows_for_derived_surface(rows, comparison)
    all_records = []

    print("")
    print("=" * 100)
    print(f"Running comparison: {comparison['name']}")
    print("=" * 100)

    if comparison["type"] == "regional":
        matrices = {}

        for region in comparison["regions"]:
            matrices[region] = build_matrix(model_cfg, var_cfg, comparison, rows, region)
            all_records.extend(
                matrix_to_records(
                    model_cfg,
                    comparison,
                    region,
                    "global 0.25 degree",
                    rows,
                    matrices[region],
                )
            )

        outpng = outdir / comparison["output_png"]
        plot_regional(comparison, rows, group_breaks, matrices, outpng)

    elif comparison["type"] == "single_domain":
        domain = comparison.get("domain", "conus")
        resolution = comparison.get("resolution", "")

        matrix = build_matrix(model_cfg, var_cfg, comparison, rows, domain)
        all_records.extend(
            matrix_to_records(model_cfg, comparison, domain, resolution, rows, matrix)
        )

        outpng = outdir / comparison["output_png"]
        plot_single_domain(comparison, rows, group_breaks, matrix, outpng)

    else:
        raise ValueError(comparison["type"])

    outcsv = outdir / comparison["output_csv"]
    write_values_csv(outcsv, all_records)

    print(f"Wrote plot: {outpng}")
    print(f"Wrote CSV : {outcsv}")
    print(f"Summary   : {summarize_records(all_records)}")

    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="../config")
    parser.add_argument("--comparison", default="all")
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    system_base = config_dir.parent
    outdir = Path(os.environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", system_base / "outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(config_dir / "models.yaml")
    var_cfg = load_yaml(config_dir / "variables.yaml")
    comp_cfg = load_yaml(config_dir / "comparisons.yaml")

    comparisons = comp_cfg["comparisons"]

    if args.comparison != "all":
        comparisons = [c for c in comparisons if c["name"] == args.comparison]
        if not comparisons:
            raise SystemExit(f"No comparison named {args.comparison}")

    all_records = []

    for comparison in comparisons:
        records = run_comparison(model_cfg, var_cfg, comparison, outdir)
        all_records.extend(records)

    all_csv = outdir / "all_scorecard_values_long.csv"
    write_values_csv(all_csv, all_records)

    print("")
    print("=" * 100)
    print("All requested comparisons complete.")
    print(f"Combined CSV: {all_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
