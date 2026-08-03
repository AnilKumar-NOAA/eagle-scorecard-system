Place the aligned scorecard NetCDF input folders here.

Expected model subdirectories:

- `aifs_2025`
- `aigfs_2025`
- `ecmwf_ifs_2025`
- `gfs_2025`
- `hrrr_2025`
- `nested_eagle_global_2025`
- `nested_eagle_lam_2025`

`./run_all_scorecards.sh` reads this directory by default. To use a different
input location without editing YAML or Python files, set:

```bash
SCORECARD_SYSTEM_DATA_DIR=/path/to/input_data ./run_all_scorecards.sh
```
