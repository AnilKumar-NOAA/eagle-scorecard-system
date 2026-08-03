#!/usr/bin/env python3

from pathlib import Path
import argparse
import yaml


def read_yaml(path, required=True, default=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return default
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(config_dir):
    config_dir = Path(config_dir)

    return {
        "models": read_yaml(config_dir / "models.yaml"),
        "variables": read_yaml(config_dir / "variables.yaml", required=False, default={}),
        "variables_v2": read_yaml(config_dir / "variables_v2.yaml", required=False, default={}),
        "comparisons": read_yaml(config_dir / "comparisons.yaml", required=False, default={"comparisons": []}),
        "plot_style": read_yaml(config_dir / "plot_style.yaml", required=False, default={}),
        "regions": read_yaml(config_dir / "regions.yaml", required=False, default={}),
        "outputs": read_yaml(config_dir / "outputs.yaml", required=False, default={}),
        "violin_plots": read_yaml(config_dir / "violin_plots.yaml", required=False, default={}),
    }


def print_summary(cfg):
    models = cfg["models"].get("models", {})
    comparisons = cfg["comparisons"].get("comparisons", [])
    variables_v2 = cfg.get("variables_v2", {})

    print("")
    print("Configuration summary")
    print("=" * 80)

    print(f"Models: {len(models)}")
    for key, info in models.items():
        print(f"  {key:24s} label={info.get('label')} color={info.get('color')}")

    print("")
    print(f"Comparisons: {len(comparisons)}")
    for item in comparisons:
        print(
            f"  {item.get('name'):35s} "
            f"{item.get('target_model')} vs {item.get('reference_model')} "
            f"color_limit={item.get('color_limit')}"
        )

    print("")
    print("Variables v2")
    print(f"  surface: {len(variables_v2.get('surface', []))}")
    print(f"  upper  : {len(variables_v2.get('upper', []))}")

    print("")
    style = cfg.get("plot_style", {})
    print("Plot style")
    print(f"  heatmap color_limit: {style.get('heatmap', {}).get('color_limit')}")
    print(f"  heatmap ticks      : {style.get('heatmap', {}).get('colorbar_ticks')}")


def validate_paths(cfg):
    data_base = Path(cfg["models"]["data_base"])
    outputs = cfg.get("outputs", {})
    output_dir = Path(outputs.get("output_dir", "."))

    print("")
    print("Path validation")
    print("=" * 80)
    print(f"data_base : {data_base} exists={data_base.exists()}")
    print(f"output_dir: {output_dir} exists={output_dir.exists()}")

    models = cfg["models"].get("models", {})
    for key, info in models.items():
        d = data_base / info.get("directory", "")
        print(f"  {key:24s} {d.name:32s} exists={d.exists()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="../config")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--validate-paths", action="store_true")
    args = parser.parse_args()

    cfg = load_all_configs(args.config_dir)

    if args.summary:
        print_summary(cfg)

    if args.validate_paths:
        validate_paths(cfg)


if __name__ == "__main__":
    main()
