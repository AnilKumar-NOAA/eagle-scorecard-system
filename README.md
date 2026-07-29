# Scorecard System

Python scripts and YAML configuration for generating scorecard-style RMSE and
bias comparison plots.

## Quick Start

```bash
python -m pip install -r requirements.txt
./run_all_scorecards.sh
```

That one command runs the legacy comparison plots and then the
configured performance heatmaps and violin plots. Generated plots and CSVs are
written to `outputs/`, which is ignored by Git.

To run only the original comparison scorecards:

```bash
python scripts/run_scorecard.py --config-dir config --comparison all --skip-performance-plots
```

The heatmap and violin scripts can still be run directly when you want to debug
or iterate on only one plot type:

```bash
python scripts/performance_heatmap.py config/performance_heatmap.yaml
python scripts/performance_violin.py config/performance_violin.yaml
```

## Flexible Plot Selection

The heatmap and violin YAML files are designed for flexible subsetting:

- Date filters: `start_date`, `end_date`, `years`, `months`
- Lead-time filters: `forecast_hours` for heatmaps, `lead_hours` and `lead_step` for violins
- Variable filters: edit `variables_v2`, set `enabled: false`, or change upper-air `levels`
- Plot filters: each item under `plots:` can override regions, models, dates, lead times, and variables

For example, a single heatmap plot can select only 2m temperature plus 500/850 hPa temperature by defining:

```yaml
variables_v2:
  surface:
    - {key: 2m_temperature, label: 2m Temperature, units: K, enabled: true}
  upper:
    - {key: temperature, label: Temperature, units: K, levels: [500, 850], enabled: true}
```

## Sample Data

The repository includes a tiny synthetic fixture at:

```text
sample_data/aligned_20250225_20251231
```

`config/models.yaml` points to this fixture by default so the config-driven
runner works in a fresh checkout. The sample NetCDF values are only for smoke
testing and are not meteorological verification results.

## Production Data

For real runs, update `data_base` in `config/models.yaml` to point at the
production aligned NetCDF directory. Also update `input_path` in
`config/performance_heatmap.yaml` and `config/performance_violin.yaml` when
using the standalone heatmap and violin plot scripts.
