from pathlib import Path
import os

import numpy as np
import xarray as xr


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
        elif "level" in da.dims:
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
