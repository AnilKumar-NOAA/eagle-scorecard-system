#!/usr/bin/env python3

from pathlib import Path
import xarray as xr

# ----------------------------------------------------------------------
# scorecard_system output control
# All figures, CSVs, and text products from this script are written here.
# Input data remain under the shared scorecard data directory.
# ----------------------------------------------------------------------
import os as _scorecard_os
from pathlib import Path as _scorecard_Path
SCORECARD_SYSTEM_OUTPUT_DIR = _scorecard_Path(_scorecard_os.environ.get(
    "SCORECARD_SYSTEM_OUTPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "outputs")
))
SCORECARD_SYSTEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BASE = Path(__import__("os").environ.get("SCORECARD_SYSTEM_DATA_DIR", Path(__file__).resolve().parents[1] / "input_data"))
AIGFS_DIR = BASE / "aigfs_2025"
PATTERN = "rmse.convobs.global.nc"

files = sorted(AIGFS_DIR.glob(PATTERN))
if not files:
    raise SystemExit(f"No AIGFS files found: {AIGFS_DIR}/{PATTERN}")

f = files[0]
print(f"Using file: {f}")

try:
    ds = xr.open_dataset(f, engine="netcdf4")
except Exception:
    ds = xr.open_dataset(f)

print("\nAll data variables:")
for v in ds.data_vars:
    da = ds[v]
    print(f"  {v:30s} dims={da.dims} shape={da.shape}")

print("\nHumidity-related candidates:")
keywords = ["humidity", "specific", "q", "sh"]
for v in ds.data_vars:
    vl = v.lower()
    if any(k in vl for k in keywords):
        da = ds[v]
        is_surface = "level" not in da.dims
        print(
            f"  {v:30s} dims={da.dims} shape={da.shape} "
            f"surface_candidate={is_surface}"
        )

print("\nSurface/no-level variables:")
for v in ds.data_vars:
    da = ds[v]
    if "level" not in da.dims:
        print(f"  {v:30s} dims={da.dims} shape={da.shape}")

ds.close()
