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

Most user selections live in one file:

```text
config/scorecard.yaml
```

Change that file to set the input directory, output directory, log directory,
enabled plot groups, model list, regions, dates, months, years, and optional
variable subset. Then run the same command:

```bash
./run_all_scorecards.sh
```

You can also pass an alternate top-level config:

```bash
./run_all_scorecards.sh /path/to/my_scorecard.yaml
```

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

For community users, start with `config/scorecard.yaml`:

- Date filters: `start_date`, `end_date`, `years`, `months`
- Model filters: edit `selection.models`
- Region filters: edit `selection.regions`
- Plot groups: set any `plot_groups.<name>.enabled` value to `false`
- Variable filters: edit `variables_v2`, set `enabled: false`, or change upper-air `levels`

The runner creates normalized runtime YAML under `logs/runtime_config/` and uses
that for the scripts. The plot-specific YAML files remain available for advanced
debugging and preserve the current figure formats.

For example, to select only 2m temperature plus 500/850 hPa temperature, add this
under `selection:` in `config/scorecard.yaml`:

```yaml
variables_v2:
  surface:
    - {key: 2m_temperature, label: 2m Temperature, units: K, enabled: true}
  upper:
    - {key: temperature, label: Temperature, units: K, levels: [500, 850], enabled: true}
```

## Input Data

The scorecard reads aligned NetCDF inputs from:

```text
input_data
```

The repository keeps lightweight data documentation only. Put aligned NetCDF
input data under `input_data`, or point the runner at another location with
`paths.input_dir` in `config/scorecard.yaml`.

## Production Data

For real runs, set `paths.input_dir` to the production aligned NetCDF directory.
The top-level YAML controls the common comparisons, date filters, model
selections, variable selections, plot groups, and output location.
