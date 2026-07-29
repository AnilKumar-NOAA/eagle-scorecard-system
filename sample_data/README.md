# Sample Scorecard Data

This folder contains a tiny synthetic fixture that mirrors the aligned scorecard
input layout:

```text
sample_data/aligned_20250225_20251231/
  nested_eagle_global_2025/
  nested_eagle_lam_2025/
  gfs_2025/
  aigfs_2025/
  ecmwf_ifs_2025/
  aifs_2025/
  hrrr_2025/
```

The NetCDF values are synthetic and are only intended for smoke testing the
scorecard scripts in a fresh checkout. They are not meteorological verification
results.

The default `config/models.yaml` points to this sample dataset so the
config-driven runner can be exercised without the large local production data.
