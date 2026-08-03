#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import os
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import load_all_configs
from violin_data import load_values


def summarize(vals):
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(vals.size),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def model_color(cfg, model_key):
    return cfg["models"]["models"][model_key].get("color", "gray")


def model_label(model_key, region):
    if model_key == "nested_eagle_global" and region == "global":
        return "Nested-EAGLE\n(Global)"
    if model_key == "nested_eagle_global" and region == "conus":
        return "Nested-EAGLE\n(CONUS)"
    if model_key == "gfs":
        return "GFS"
    if model_key == "aigfs":
        return "AIGFS"
    if model_key == "aifs":
        return "AIFS"
    if model_key == "hrrr":
        return "HRRR"
    return model_key


def region_label(cfg, region):
    return cfg.get("regions", {}).get("regions", {}).get(region, {}).get("label", region.upper())


def enabled_variable_levels(cfg):
    out = []

    for item in cfg.get("variables_v2", {}).get("surface", []):
        if item.get("enabled", True):
            out.append((item["key"], None))

    for item in cfg.get("variables_v2", {}).get("upper", []):
        if item.get("enabled", True):
            for lev in item.get("levels", []):
                out.append((item["key"], int(lev)))

    return out


def load_all_response_values(cfg, model_key, models, metric, region, lead_start, lead_end):
    chunks = []

    for variable, level in enabled_variable_levels(cfg):
        raw_by_model = {}

        for m in models:
            vals = load_values(
                cfg=cfg,
                model_key=m,
                metric=metric,
                variable=variable,
                region=region,
                level=level,
                lead_start=lead_start,
                lead_end=lead_end,
            )

            vals = vals[np.isfinite(vals)]

            if vals.size > 0:
                raw_by_model[m] = vals

        # Require all selected models to have data for this variable/level.
        # Then force equal sample count across models so one model does not
        # dominate the all-response distribution only because it has more values.
        if any(m not in raw_by_model for m in models):
            continue

        min_count = min(raw_by_model[m].size for m in models)

        if min_count < 2:
            continue

        equal_raw = {}

        for m in models:
            vals = raw_by_model[m]
            vals = vals[np.isfinite(vals)]

            # Use first min_count values. This assumes load_values returns values
            # in consistent lead/init/order for each model.
            equal_raw[m] = vals[:min_count]

        pooled = np.concatenate([equal_raw[m] for m in models])
        pooled = pooled[np.isfinite(pooled)]

        if pooled.size == 0:
            continue

        denom = np.nanmedian(pooled)

        if not np.isfinite(denom) or denom <= 0:
            continue

        norm_vals = equal_raw[model_key] / denom
        norm_vals = norm_vals[np.isfinite(norm_vals)]

        if norm_vals.size:
            chunks.append(norm_vals)

    if not chunks:
        return np.array([], dtype=float)

    return np.concatenate(chunks)


def clipped_plot_data(data, good):
    all_vals = np.concatenate([data[i] for i in good])
    all_vals = all_vals[np.isfinite(all_vals)]

    if all_vals.size == 0:
        return [], 1.0

    cap = float(np.nanpercentile(all_vals, 98.5))
    cap = max(cap, 1.25)

    plot_data = []

    for i in good:
        vals = data[i]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0.0) & (vals <= cap)]

        if vals.size < 2:
            vals = data[i][np.isfinite(data[i])]

        plot_data.append(vals)

    return plot_data, cap


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_one(cfg, plot_cfg):
    metric = plot_cfg.get("metric", "rmse")
    regions = plot_cfg["regions"]
    models = plot_cfg["models"]
    lead_start, lead_end = plot_cfg.get("lead_hours", [24, 240])

    output_dir = Path(os.environ.get("SCORECARD_SYSTEM_OUTPUT_DIR", cfg.get("outputs", {}).get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg.get("outputs", {}).get("dpi", 200))
    outpng = output_dir / plot_cfg["output"]
    outcsv = output_dir / f"{plot_cfg['name']}_summary.csv"

    fig, axes = plt.subplots(
        len(regions),
        1,
        figsize=(11.5, 4.9 * len(regions) + 1.2),
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
            vals = load_all_response_values(
                cfg=cfg,
                model_key=model_key,
                models=models,
                metric=metric,
                region=region,
                lead_start=int(lead_start),
                lead_end=int(lead_end),
            )

            s = summarize(vals)

            print(
                f"{plot_cfg['name']} | {region:8s} | {model_key:22s} "
                f"Count={s['count']} Median={s['median']:.3f} Mean={s['mean']:.3f}"
            )

            data.append(vals)
            labels.append(model_label(model_key, region))
            colors.append(model_color(cfg, model_key))
            stats.append(s)

            rows.append(
                {
                    "plot": plot_cfg["name"],
                    "region": region,
                    "model": model_key,
                    "metric": metric,
                    "value_type": "normalized_rmse_all_responses",
                    **s,
                }
            )

        positions = np.arange(1, len(models) + 1)
        good = [i for i, vals in enumerate(data) if vals.size > 1]

        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.55, zorder=0)

        if good:
            plot_data, cap = clipped_plot_data(data, good)

            vp = ax.violinplot(
                plot_data,
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
                plot_data,
                positions=[positions[i] for i in good],
                widths=0.15,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.1},
                capprops={"color": "black", "linewidth": 1.1},
            )

            ax.set_ylim(0, cap * 1.75)

            for i in good:
                x = positions[i]
                s = stats[i]

                ax.scatter(
                    [x],
                    [min(s["mean"], cap)],
                    s=48,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=6,
                )

                ax.text(
                    x,
                    0.955,
                    f"Count = {s['count']}\nMedian = {s['median']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
                )

                ax.text(
                    x,
                    0.825,
                    f"{s['mean']:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
                )

        ax.set_title(region_label(cfg, region), fontsize=17, fontweight="bold", pad=6)
        ax.set_ylabel("Normalized RMSE\nLower is Better", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)

        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
            tick.set_fontweight("bold")

        ax.grid(True, axis="y", linestyle=":", alpha=0.28)
        ax.set_xlim(0.45, len(models) + 0.55)

    fig.suptitle(plot_cfg["title"], fontsize=20, fontweight="bold", y=0.990)
    fig.text(0.5, 0.925, plot_cfg["subtitle"], ha="center", fontsize=12.5)

    fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.890])
    fig.savefig(outpng, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    write_csv(outcsv, rows)

    print(f"Wrote {outpng}")
    print(f"Wrote {outcsv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--plot-config", default="config/all_response_violin_conus_hrrr_48h.yaml")
    args = parser.parse_args()

    cfg = load_all_configs(args.config_dir)

    with open(args.plot_config, "r") as f:
        plot_file = yaml.safe_load(f)

    for plot_cfg in plot_file.get("plots", []):
        plot_one(cfg, plot_cfg)


if __name__ == "__main__":
    main()
