#!/bin/bash
set -e

SYSTEM_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DATA="${SCORECARD_SYSTEM_DATA_DIR:-$SYSTEM_BASE/data/new_data}"
SCRIPT_DIR=$SYSTEM_BASE/scripts
OUTPUT_DIR="${SCORECARD_SYSTEM_OUTPUT_DIR:-$SYSTEM_BASE/outputs}"
LOG_DIR="${SCORECARD_SYSTEM_LOG_DIR:-$SYSTEM_BASE/logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export SCORECARD_SYSTEM_DATA_DIR="$SYSTEM_DATA"
export SCORECARD_SYSTEM_OUTPUT_DIR="$OUTPUT_DIR"

echo "SYSTEM_BASE=$SYSTEM_BASE"
echo "SYSTEM_DATA=$SYSTEM_DATA"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "$SCRIPT_DIR"

echo ""
echo "Running config-driven scorecard system..."
python -m py_compile run_scorecard.py
python run_scorecard.py --config-dir ../config --comparison all | tee "$LOG_DIR/run_config_driven_scorecards.log"


echo ""
echo "Running required scorecard-style RMSE scripts..."
for s in \
  plots_global_4models_plot_scorecard_style_upper_multilevel_rmse_0_48h.py \
  plots_global_4models_plot_scorecard_style_surface_rmse_heatmap_0_48h.py
do
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s"
  python "$s" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Running global 4-model panel scripts..."
for s in plots_global_4models_*.py; do
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s"
  python "$s" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Running Nested-LAM/HRRR/GFS 0-48 h scripts..."
for s in nested_lam_hrrr_gfs_0_48_*.py; do
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s"
  python "$s" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Running Nested-LAM/HRRR/GFS CONUS D1-D10 scripts..."
for s in nested_lam_hrrr_gfs_conus_D1_D10_*.py; do
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s"
  python "$s" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Running GFS difference plots..."
for s in plot_rmse_bias_difference_from_gfs_conus_D1_D10.py; do
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s"
  python "$s" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Running violin plot scripts..."
for entry in \
  "plot_violin_forecast_skill.py --config-dir ../config" \
  "plot_violin_all_responses.py --config-dir ../config --plot-config ../config/all_response_violin.yaml" \
  "plot_violin_all_responses_no_hrrr.py --config-dir ../config --plot-config ../config/all_response_violin_no_hrrr.yaml" \
  "plot_violin_all_responses_conus_hrrr_48h.py --config-dir ../config --plot-config ../config/all_response_violin_conus_hrrr_48h.yaml"
do
  set -- $entry
  s="$1"
  shift
  [ -f "$s" ] || continue

  echo ""
  echo "Compiling $s"
  python -m py_compile "$s"

  echo "Running $s $*"
  python "$s" "$@" | tee "$LOG_DIR/${s%.py}.log"
done

echo ""
echo "Finished."
echo "Outputs:"
ls -lh "$OUTPUT_DIR"
