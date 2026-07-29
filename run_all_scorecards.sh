#!/usr/bin/env bash
set -euo pipefail

SYSTEM_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCORECARD_CONFIG_DIR:-$SYSTEM_BASE/config}"
OUTPUT_DIR="${SCORECARD_OUTPUT_DIR:-$SYSTEM_BASE/outputs}"
LOG_DIR="${SCORECARD_LOG_DIR:-$SYSTEM_BASE/logs}"
COMPARISON="${SCORECARD_COMPARISON:-all}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export SCORECARD_SYSTEM_DATA_DIR="${SCORECARD_SYSTEM_DATA_DIR:-$SYSTEM_BASE/sample_data/aligned_20250225_20251231}"
export SCORECARD_SYSTEM_OUTPUT_DIR="$OUTPUT_DIR"

echo "SYSTEM_BASE=$SYSTEM_BASE"
echo "CONFIG_DIR=$CONFIG_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "COMPARISON=$COMPARISON"
echo "SCORECARD_SYSTEM_DATA_DIR=$SCORECARD_SYSTEM_DATA_DIR"

echo ""
echo "Running scorecard system..."
python -m py_compile "$SYSTEM_BASE"/scripts/*.py
python "$SYSTEM_BASE/scripts/run_scorecard.py" \
  --config-dir "$CONFIG_DIR" \
  --comparison "$COMPARISON" \
  2>&1 | tee "$LOG_DIR/run_all_scorecards.log"

if [[ "${RUN_EXTRA_LEGACY_SCRIPTS:-0}" == "1" ]]; then
  echo ""
  echo "Running extra legacy plotting scripts..."
  cd "$SYSTEM_BASE/scripts"

  for s in \
    plots_global_4models_plot_scorecard_style_upper_multilevel_rmse_0_48h.py \
    plots_global_4models_plot_scorecard_style_surface_rmse_heatmap_0_48h.py \
    plots_global_4models_*.py \
    nested_lam_hrrr_gfs_0_48_*.py \
    nested_lam_hrrr_gfs_conus_D1_D10_*.py
  do
    [[ -f "$s" ]] || continue
    echo ""
    echo "Running $s"
    python -m py_compile "$s"
    python "$s" 2>&1 | tee "$LOG_DIR/${s%.py}.log"
  done
fi

echo ""
echo "Finished."
echo "Outputs:"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print | sort
