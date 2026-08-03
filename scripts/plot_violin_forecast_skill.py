#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import os
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import load_all_configs


def open_ds(path):
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return xr.open_dataset(path)


def get_pattern(model_info, metric, region):
    if region == "global":
        pattern = model_info.get("global_pattern")
        if pattern is None:
            patterns = model_info.get("global_patterns", [])
            pattern = patterns[0] if patterns else None
    else:
        pattern = model_info.get("regional_pattern")
        if pattern is None:
            patterns = model_info.get("regional_patterns", [])
            pattern = patterns[0] if patterns else None

    if pattern is None:
        pattern = model_info.get("default_pattern")
        if pattern is None:
            patterns = model_info.get("default_patterns", [])
            pattern = patterns[0] if patterns else None

    if pattern is None:
        raise ValueError(f"No file pattern for {model_info.get('label')} region={region}")

    return pattern.format(metric=metric, region=region)


def find_fhr_name(da):
    for name in ["fhr", "lead", "lead_time", "forecast_hour", "forecast_hours"]:
        if name in da.coords or name in da.dims:
            return name
    return None


def fhr_to_hours(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return (arr / np.timedelta64(1, "h")).astype(int)
    return arr.astype(int)


def load_values(cfg, model_key, metric, variable, region, level, lead_start, lead_end):
    models_cfg = cfg["models"]
    data_base = Path(os.environ.get("SCORECARD_SYSTEM_DATA_DIR", models_cfg["data_base"]))
    model_info = models_cfg["models"][model_key]
    model_dir = data_base / model_info["directory"]

    pattern = get_pattern(model_info, metric, region)
    files = sorted(model_dir.glob(pattern))

    if not files:
        print(f"WARNING no files: {model_key} {region} {model_dir}/{pattern}")
        return np.array([], dtype=float)

    out = []

    for f in files:
        ds = open_ds(f)

        if variable not in ds.data_vars:
            print(f"WARNING variable missing: {variable} in {f}")
            ds.close()
            continue

        da = ds[variable]

        fhr_name = find_fhr_name(da)
        if fhr_name is not None:
            fhrs = fhr_to_hours(da[fhr_name].values)
            idx = np.where((fhrs >= int(lead_start)) & (fhrs <= int(lead_end)))[0]
            if idx.size == 0:
                ds.close()
                continue
            da = da.isel({fhr_name: idx})

        if level is not None:
            if "level" not in da.coords and "level" not in da.dims:
                print(f"WARNING level requested but missing: {f.name}")
                ds.close()
                continue

            levels = np.asarray(da["level"].values).astype(int)
            idx = np.where(levels == int(level))[0]

            if idx.size == 0:
                print(f"WARNING level {level} missing in {f.name}; available={levels}")
                ds.close()
                continue

            da = da.isel(level=idx[0])
        else:
            if "level" in da.dims:
                ds.close()
                continue

        vals = np.asarray(da.values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size:
            out.append(vals)

        ds.close()

    if not out:
        return np.array([], dtype=float)

    return np.concatenate(out)


def summarize(vals):
    if vals.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, min=np.nan, max=np.nan)

    return dict(
        n=int(vals.size),
        mean=float(np.nanmean(vals)),
        median=float(np.nanmedian(vals)),
        std=float(np.nanstd(vals)),
        min=float(np.nanmin(vals)),
        max=float(np.nanmax(vals)),
    )


def model_label(cfg, model_key, region=None):
    info = cfg["models"]["models"][model_key]

    if model_key == "nested_eagle_global" and region == "global":
        return "Nested-EAGLE\n(Global)"

    if model_key == "nested_eagle_global" and region == "conus":
        return "Nested-EAGLE\n(CONUS)"

    label = info.get("plot_label", info.get("label", model_key))
    sub = info.get("plot_sublabel", "")

    return f"{label}\n{sub}" if sub else label


def model_color(cfg, model_key):
    return cfg["models"]["models"][model_key].get("color", "gray")


def region_label(cfg, region):
    return cfg.get("regions", {}).get("regions", {}).get(region, {}).get("label", region.upper())


def variable_label(cfg, variable, level):
    for group in ["surface", "upper"]:
        for item in cfg.get("variables_v2", {}).get(group, []):
            if item.get("key") == variable:
                label = item.get("label", variable)
                units = item.get("units", "")
                if level is not None:
                    label = f"{label} {level} hPa"
                return label, units

    if level is not None:
        return f"{variable} {level} hPa", ""

    return variable, ""


def write_summary_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_one(cfg, plot_cfg):
    metric = plot_cfg.get("metric", "rmse")
    variable = plot_cfg["variable"]
    level = plot_cfg.get("level")
    models = plot_cfg["models"]
    regions = plot_cfg["regions"]
    lead_start, lead_end = plot_cfg.get("lead_hours", [24, 240])

    output_dir = Path(os.environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", cfg.get("outputs", {}).get("output_dir", "../outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg.get("outputs", {}).get("dpi", 200))
    outpng = output_dir / plot_cfg.get("output", f"{plot_cfg['name']}.png")
    outcsv = output_dir / f"{plot_cfg['name']}_summary.csv"

    var_label, units = variable_label(cfg, variable, level)
    ylabel = f"{metric.upper()} ({units})\nLower is Better" if units else f"{metric.upper()}\nLower is Better"

    fig_height = 4.2 * len(regions) + 1.4
    fig, axes = plt.subplots(
        len(regions),
        1,
        figsize=(11.5, fig_height),
        squeeze=False,
    )
    axes = axes[:, 0]

    rows = []

    for ax, region in zip(axes, regions):
        data = []
        labels = []
        colors = []
        stats = []

        for model_key in models:
            vals = load_values(
                cfg=cfg,
                model_key=model_key,
                metric=metric,
                variable=variable,
                region=region,
                level=level,
                lead_start=lead_start,
                lead_end=lead_end,
            )

            s = summarize(vals)

            print(
                f"{plot_cfg['name']} | {region:8s} | {model_key:22s} "
                f"Count={s['n']} Median={s['median']:.4g} Mean={s['mean']:.4g}"
            )

            data.append(vals)
            labels.append(model_label(cfg, model_key, region))
            colors.append(model_color(cfg, model_key))
            stats.append(s)

            rows.append(
                {
                    "plot": plot_cfg["name"],
                    "region": region,
                    "model": model_key,
                    "metric": metric,
                    "variable": variable,
                    "level": level,
                    **s,
                }
            )

        positions = np.arange(1, len(models) + 1)
        good = [i for i, vals in enumerate(data) if vals.size > 1]

        if good:
            vp = ax.violinplot(
                [data[i] for i in good],
                positions=[positions[i] for i in good],
                widths=0.62,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body, i in zip(vp["bodies"], good):
                body.set_facecolor(colors[i])
                body.set_edgecolor("black")
                body.set_alpha(0.78)
                body.set_linewidth(1.2)

            ax.boxplot(
                [data[i] for i in good],
                positions=[positions[i] for i in good],
                widths=0.15,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.1},
                capprops={"color": "black", "linewidth": 1.1},
            )

            all_finite = np.concatenate([data[i] for i in good])
            ymin = float(np.nanmin(all_finite))
            ymax = float(np.nanmax(all_finite))
            span = max(ymax - ymin, 1.0)

            ax.set_ylim(max(0.0, ymin - 0.05 * span), ymax + 0.38 * span)

            for i in good:
                x = positions[i]
                s = stats[i]

                ax.scatter(
                    [x],
                    [s["mean"]],
                    s=48,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=6,
                )

                ax.text(
                    x,
                    0.965,
                    f"Count = {s['n']}\nMedian = {s['median']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                )

                ax.text(
                    x,
                    0.835,
                    f"{s['mean']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                )

        for i, vals in enumerate(data):
            if vals.size <= 1:
                ax.text(positions[i], 0.5, "No data", ha="center", va="center", fontsize=10)

        ax.set_title(region_label(cfg, region), fontsize=17, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)

        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
            tick.set_fontweight("bold")

        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.set_xlim(0.45, len(models) + 0.55)

    fig.suptitle(
        plot_cfg.get("title", "Forecast RMSE Guidance: All Days"),
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.942,
        f"{plot_cfg.get('subtitle', var_label)} | Forecast Hours {lead_start}-{lead_end}",
        ha="center",
        fontsize=14,
    )

    fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.92])
    fig.savefig(outpng, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    write_summary_csv(outcsv, rows)

    print(f"Wrote {outpng}")
    print(f"Wrote {outcsv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="../config")
    parser.add_argument("--plot", default="all")
    args = parser.parse_args()

    cfg = load_all_configs(args.config_dir)
    plots = cfg.get("violin_plots", {}).get("plots", [])

    if not plots:
        raise SystemExit("No plots found in config/violin_plots.yaml")

    ran = 0

    for plot_cfg in plots:
        if args.plot != "all" and plot_cfg["name"] != args.plot:
            continue

        plot_one(cfg, plot_cfg)
        ran += 1

    if ran == 0:
        raise SystemExit(f"No matching plot found for --plot {args.plot}")


if __name__ == "__main__":
    main()
