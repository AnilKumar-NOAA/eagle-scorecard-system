#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import shlex
from pathlib import Path
from typing import Any

import yaml


DATE_KEYS = ("start_date", "end_date", "years", "months")


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False)


def resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def selected_names(value: Any) -> set[str] | None:
    if value in (None, "all"):
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def filter_named(items: list[dict[str, Any]], names: Any) -> list[dict[str, Any]]:
    wanted = selected_names(names)
    if wanted is None:
        return items
    return [item for item in items if item.get("name") in wanted]


def apply_overrides(items: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    updated = []
    for item in items:
        merged = copy.deepcopy(item)
        merged.update(overrides.get(str(item.get("name")), {}))
        updated.append(merged)
    return updated


def apply_global_selection(config: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key in DATE_KEYS:
        if key in selection:
            out[key] = selection[key]
    if "variables_v2" in selection:
        out["variables_v2"] = selection["variables_v2"]
    if "regions" in selection:
        out["regions"] = selection["regions"]
    return out


def apply_plot_selection(config: dict[str, Any], names: Any) -> dict[str, Any]:
    out = copy.deepcopy(config)
    if "plots" in out:
        out["plots"] = filter_named(out["plots"], names)
    return out


def selected_plots(plots: list[dict[str, Any]], names: Any) -> list[dict[str, Any]]:
    return filter_named(copy.deepcopy(plots), names)


def variables_v2_to_variables_yaml(base: dict[str, Any], variables_v2: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    row_groups: list[dict[str, Any]] = []

    for item in variables_v2.get("upper", []):
        if not item.get("enabled", True):
            continue
        key = item["key"]
        rows = [
            {"var": key, "label": f"{key} {int(level)}", "levels": [int(level)]}
            for level in item.get("levels", [])
        ]
        if rows:
            row_groups.append({"name": key, "rows": rows})

    surface_rows = []
    for item in variables_v2.get("surface", []):
        if not item.get("enabled", True):
            continue
        key = item["key"]
        surface_rows.append({"var": key, "label": key, "levels": None})
    if surface_rows:
        row_groups.append({"name": "surface", "rows": surface_rows})

    if row_groups:
        out["row_groups"] = row_groups
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/scorecard.yaml")
    parser.add_argument("--print-env", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    master_path = resolve_path(repo, args.config)
    master = read_yaml(master_path)

    advanced = master.get("advanced", {})
    config_dir = resolve_path(repo, advanced.get("config_dir", "config"))
    runtime_dir = resolve_path(repo, advanced.get("runtime_config_dir", "logs/runtime_config"))
    paths = master.get("paths", {})
    selection = master.get("selection", {})
    plot_groups = master.get("plot_groups", {})

    input_dir = resolve_path(repo, paths.get("input_dir", "input_data"))
    output_dir = resolve_path(repo, paths.get("output_dir", "outputs"))
    log_dir = resolve_path(repo, paths.get("log_dir", "logs"))

    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = read_yaml(config_dir / "models.yaml")
    models_cfg["data_base"] = str(input_dir)
    enabled_models = selected_names(selection.get("models"))
    if enabled_models is not None:
        models_cfg["models"] = {
            key: value for key, value in models_cfg.get("models", {}).items() if key in enabled_models
        }
    write_yaml(runtime_dir / "models.yaml", models_cfg)

    for name in ("variables_v2.yaml", "regions.yaml", "outputs.yaml", "plot_style.yaml"):
        source = config_dir / name
        if source.exists():
            write_yaml(runtime_dir / name, read_yaml(source))

    variables_cfg = read_yaml(config_dir / "variables.yaml")
    if "variables_v2" in selection:
        variables_cfg = variables_v2_to_variables_yaml(variables_cfg, selection["variables_v2"])
    write_yaml(runtime_dir / "variables.yaml", variables_cfg)

    config_scorecards = plot_groups.get("config_scorecards", {})
    comparisons = {"comparisons": copy.deepcopy(config_scorecards.get("plots", []))}
    comparisons["comparisons"] = filter_named(
        comparisons.get("comparisons", []),
        config_scorecards.get("comparisons", "all"),
    )
    for comparison in comparisons["comparisons"]:
        for key in DATE_KEYS:
            if key in selection:
                comparison[key] = selection[key]
        if "regions" in selection and comparison.get("type") == "regional":
            comparison["regions"] = selection["regions"]
    write_yaml(runtime_dir / "comparisons.yaml", comparisons)

    violin_group = plot_groups.get("violin", {})
    violin_plots = selected_plots(
        violin_group.get("plots", []),
        violin_group.get("selected", "all"),
    )
    for name in ("performance_heatmap.yaml", "performance_violin.yaml"):
        cfg = apply_global_selection(read_yaml(config_dir / name), selection)
        cfg["input_path"] = str(input_dir)
        cfg["output_path"] = str(output_dir)
        cfg["require_exact_time_match"] = advanced.get("require_exact_time_match", True)
        if name == "performance_violin.yaml":
            cfg["plots"] = copy.deepcopy(violin_plots)
        write_yaml(runtime_dir / name, cfg)

    violin_runtime_files = {
        "plot_violin_all_responses_no_hrrr.py": "all_response_violin_no_hrrr.yaml",
        "plot_violin_all_responses_conus_hrrr_48h.py": "all_response_violin_conus_hrrr_48h.yaml",
    }
    for script_name, runtime_name in violin_runtime_files.items():
        cfg = {"plots": [copy.deepcopy(plot) for plot in violin_plots if plot.get("script") == script_name]}
        for plot in cfg.get("plots", []):
            plot.pop("script", None)
            for key in DATE_KEYS:
                if key in selection:
                    plot[key] = selection[key]
        write_yaml(runtime_dir / runtime_name, cfg)

    enabled = {key: bool(value.get("enabled", True)) for key, value in plot_groups.items()}
    runner_cfg = {
        "enabled": enabled,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "runtime_config_dir": str(runtime_dir),
        "config_scorecard_comparison": plot_groups.get("config_scorecards", {}).get("comparisons", "all"),
    }
    write_yaml(runtime_dir / "runner.yaml", runner_cfg)

    if args.print_env:
        env = {
            "SCORECARD_SYSTEM_DATA_DIR": input_dir,
            "SCORECARD_SYSTEM_OUTPUT_DIR": output_dir,
            "SCORECARD_SYSTEM_LOG_DIR": log_dir,
            "SCORECARD_SYSTEM_RUNTIME_CONFIG_DIR": runtime_dir,
            "SCORECARD_RUN_CONFIG_SCORECARDS": str(enabled.get("config_scorecards", True)).lower(),
            "SCORECARD_RUN_SCORECARD_STYLE_RMSE": str(enabled.get("scorecard_style_rmse", True)).lower(),
            "SCORECARD_RUN_GLOBAL_4MODEL_PANELS": str(enabled.get("global_4model_panels", True)).lower(),
            "SCORECARD_RUN_NESTED_0_48H": str(enabled.get("nested_lam_hrrr_gfs_0_48h", True)).lower(),
            "SCORECARD_RUN_NESTED_D1_D10": str(enabled.get("nested_lam_hrrr_gfs_conus_d1_d10", True)).lower(),
            "SCORECARD_RUN_GFS_DIFFERENCE": str(enabled.get("gfs_difference", True)).lower(),
            "SCORECARD_RUN_VIOLIN": str(enabled.get("violin", True)).lower(),
            "SCORECARD_CONFIG_COMPARISON": runner_cfg["config_scorecard_comparison"],
        }
        for key, value in env.items():
            print(f"export {key}={shlex.quote(str(value))}")
    else:
        print(f"Wrote runtime config: {runtime_dir}")


if __name__ == "__main__":
    main()
