#!/usr/bin/env python3

from pathlib import Path
import re
import shutil
import xarray as xr
import pandas as pd

START = "2025-02-25"
END = "2025-12-31"

ROOT = Path("/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv")
DATA = ROOT / "scorecard_system" / "data"
AIFS_METRICS = ROOT / "AIFS" / "aifs_2025_outputs"

OUT_BASE = ROOT / "scorecard_system" / "outputs" / "aligned_20250225_20251231"

INPUT_DIRS = {
    "aifs_2025": AIFS_METRICS,
    "aigfs_2025": DATA / "aigfs_2025",
    "gfs_2025": DATA / "gfs_zarr_2025",
    "hrrr_2025": DATA / "hrrr_2025",
    "nested_eagle_global_2025": DATA / "nested_eagle_2025",
    "nested_eagle_lam_2025": DATA / "nested_eagle_lam_2025",
}

DATE_SUFFIX_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}\.nc$")
DATE_IN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")
WORK_DIR_RE = re.compile(r"work_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")


def canonical_name(path: Path) -> str:
    """Remove monthly date suffix so monthly files merge into one final file."""
    name = path.name
    name = DATE_SUFFIX_RE.sub(".nc", name)
    return name


def sort_key(path: Path):
    text = str(path)
    m = DATE_IN_NAME_RE.search(text)
    if m:
        return m.group(1)
    m = WORK_DIR_RE.search(text)
    if m:
        return m.group(1)
    return "0000-00-00"


def collect_groups(indir: Path):
    """
    Collect NetCDF files into merge groups.
    Prefer top-level monthly files when present.
    Use work_* files only as fallback for keys without top-level files.
    """
    top_files = [p for p in indir.glob("*.nc") if p.name != "subregions.nc"]
    rec_files = [p for p in indir.rglob("*.nc") if p.name != "subregions.nc"]

    top_groups = {}
    for f in top_files:
        key = canonical_name(f)
        top_groups.setdefault(key, []).append(f)

    work_groups = {}
    for f in rec_files:
        if f in top_files:
            continue
        key = canonical_name(f)
        work_groups.setdefault(key, []).append(f)

    groups = dict(top_groups)
    for key, files in work_groups.items():
        if key not in groups:
            groups[key] = files

    for key in groups:
        groups[key] = sorted(groups[key], key=sort_key)

    return groups


def subset_ds(ds: xr.Dataset) -> xr.Dataset:
    if "t0" in ds.coords or "t0" in ds.dims:
        return ds.sel(t0=slice(START, END))
    if "time" in ds.coords or "time" in ds.dims:
        return ds.sel(time=slice(START, END))
    return ds


def merge_datasets(dsets):
    if len(dsets) == 1:
        out = dsets[0]
    else:
        # If chunks all contain the same full t0 axis, combine_first fills NaNs.
        same_t0 = (
            all("t0" in ds.coords for ds in dsets)
            and all(ds.sizes.get("t0") == dsets[0].sizes.get("t0") for ds in dsets)
            and all((ds["t0"].values == dsets[0]["t0"].values).all() for ds in dsets)
        )

        if same_t0:
            out = dsets[0]
            for ds in dsets[1:]:
                out = out.combine_first(ds)
        else:
            dim = "t0" if "t0" in dsets[0].dims else "time" if "time" in dsets[0].dims else None
            if dim is None:
                out = dsets[0]
            else:
                out = xr.concat(dsets, dim=dim, coords="minimal", compat="override")
                out = out.sortby(dim)

                # Drop duplicate times if any duplicate month/work files slipped in.
                idx = pd.Index(out[dim].values)
                keep = ~idx.duplicated()
                out = out.isel({dim: keep})

    if "t0" in out.coords:
        out = out.sortby("t0")
    elif "time" in out.coords:
        out = out.sortby("time")

    return out


def process_model(model: str, indir: Path):
    print("\n" + "=" * 100, flush=True)
    print(f"MODEL: {model}", flush=True)
    print(f"INPUT: {indir}", flush=True)

    if not indir.exists():
        print(f"SKIP missing directory: {indir}", flush=True)
        return

    outdir = OUT_BASE / model
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sub = indir / "subregions.nc"
    if sub.exists():
        shutil.copy2(sub, outdir / "subregions.nc")

    groups = collect_groups(indir)
    print(f"Groups found: {len(groups)}", flush=True)

    for key, files in sorted(groups.items()):
        print(f"\nMerging {key}", flush=True)
        print(f"  files: {len(files)}", flush=True)

        dsets = []
        for f in files:
            ds = xr.open_dataset(f)
            ds = subset_ds(ds)

            if "t0" in ds.dims and ds.sizes["t0"] == 0:
                ds.close()
                continue
            if "time" in ds.dims and ds.sizes["time"] == 0:
                ds.close()
                continue

            dsets.append(ds)

        if not dsets:
            print(f"  SKIP no data in AIFS period for {key}", flush=True)
            continue

        merged = merge_datasets(dsets)

        out = outdir / key
        if out.exists():
            out.unlink()

        merged.to_netcdf(out)

        if "t0" in merged.sizes:
            print(f"  wrote {out.name}: t0={merged.sizes['t0']}", flush=True)
        elif "time" in merged.sizes:
            print(f"  wrote {out.name}: time={merged.sizes['time']}", flush=True)
        else:
            print(f"  wrote {out.name}", flush=True)

        for ds in dsets:
            ds.close()
        if merged not in dsets:
            merged.close()

    nout = len(list(outdir.glob("*.nc")))
    print(f"\nDONE {model}: {nout} files -> {outdir}", flush=True)


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Aligning metrics to common AIFS period: {START} through {END}", flush=True)
    print(f"Output base: {OUT_BASE}", flush=True)

    for model, indir in INPUT_DIRS.items():
        process_model(model, indir)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
