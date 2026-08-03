# Scorecard System

Python scripts and YAML configuration for generating scorecard-style RMSE and
bias comparison plots.

## Quick Start

```bash
python -m pip install -r requirements.txt
./run_all_scorecards.sh
```

That one command runs the complete scorecard suite and writes plots/CSVs to
`outputs/`. The verified reference figures in `outputs/` are tracked in Git so
the repository preserves the same figure formats generated locally.

To run only the config-driven comparison scorecards:

```bash
python scripts/run_scorecard.py --config-dir config --comparison all
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

The repository keeps lightweight data documentation only. Put aligned NetCDF
input data under `data/new_data`, or point the runner at another location with
`SCORECARD_SYSTEM_DATA_DIR=/path/to/aligned_20250225_20251231`.

## Production Data

For real runs, set `SCORECARD_SYSTEM_DATA_DIR` to the production aligned NetCDF
directory. The YAML files control the comparisons, date filters, lead-time
filters, model selections, variable selections, and output names.
