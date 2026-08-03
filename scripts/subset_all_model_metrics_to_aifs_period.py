#!/usr/bin/env python3

from pathlib import Path
import shutil
import xarray as xr

START = "2025-02-25"
END = "2025-12-31"

ROOT = Path("/scratch3/NAGAPE/epic/role-epic/EAGLE/eagle_models_vv")
DATA = ROOT / "scorecard_system" / "data"
AIFS_METRICS = ROOT / "AIFS" / "aifs_2025_outputs"

INPUT_DIRS = {
    "aifs_2025": AIFS_METRICS,
    "aigfs_2025": DATA / "aigfs_2025",
    "gfs_2025": DATA / "gfs_zarr_2025",
    "hrrr_2025": DATA / "hrrr_2025",
    "nested_eagle_global_2025": DATA / "nested_eagle_2025",
    "nested_eagle_lam_2025": DATA / "nested_eagle_lam_2025",
}

OUT_BASE = ROOT / "scorecard_system" / "outputs" / "aligned_20250225_20251231"
OUT_BASE.mkdir(parents=True, exist_ok=True)


def subset_one_file(infile: Path, outfile: Path) -> None:
    if infile.name == "subregions.nc":
        shutil.copy2(infile, outfile)
        print(f"  copied {infile.name}", flush=True)
        return

    ds = xr.open_dataset(infile)

    if "t0" in ds.coords or "t0" in ds.dims:
        ds_out = ds.sel(t0=slice(START, END))
    elif "time" in ds.coords or "time" in ds.dims:
        ds_out = ds.sel(time=slice(START, END))
    else:
        print(f"  WARNING no t0/time in {infile.name}; copying unchanged", flush=True)
        ds_out = ds

    if outfile.exists():
        outfile.unlink()

    ds_out.to_netcdf(outfile)

    if "t0" in ds_out.sizes:
        print(f"  {infile.name}: t0={ds_out.sizes['t0']}", flush=True)
    elif "time" in ds_out.sizes:
        print(f"  {infile.name}: time={ds_out.sizes['time']}", flush=True)
    else:
        print(f"  {infile.name}: no t0/time", flush=True)

    ds.close()
    if ds_out is not ds:
        ds_out.close()


def main():
    print(f"Aligning all model metrics to {START} through {END}", flush=True)
    print(f"Output base: {OUT_BASE}", flush=True)

    for model, indir in INPUT_DIRS.items():
        print("\n" + "=" * 90, flush=True)
        print(f"MODEL: {model}", flush=True)
        print(f"INPUT: {indir}", flush=True)

        if not indir.exists():
            print(f"SKIP missing input directory: {indir}", flush=True)
            continue

        outdir = OUT_BASE / model
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        files = sorted(indir.glob("*.nc"))
        if not files:
            print(f"SKIP no .nc metric files in {indir}", flush=True)
            continue

        for f in files:
            subset_one_file(f, outdir / f.name)

        print(f"DONE {model}: {len(list(outdir.glob('*.nc')))} files", flush=True)

    print("\nDONE all available models", flush=True)


if __name__ == "__main__":
    main()
