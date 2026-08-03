#!/bin/bash
set -e

SYSTEM_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCORECARD_CONFIG="${SCORECARD_CONFIG:-${1:-$SYSTEM_BASE/config/scorecard.yaml}}"
SCRIPT_DIR=$SYSTEM_BASE/scripts

eval "$(python "$SCRIPT_DIR/prepare_scorecard_config.py" --config "$SCORECARD_CONFIG" --print-env)"

SYSTEM_DATA="$SCORECARD_SYSTEM_DATA_DIR"
OUTPUT_DIR="$SCORECARD_SYSTEM_OUTPUT_DIR"
LOG_DIR="$SCORECARD_SYSTEM_LOG_DIR"
RUNTIME_CONFIG_DIR="$SCORECARD_SYSTEM_RUNTIME_CONFIG_DIR"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export SCORECARD_SYSTEM_DATA_DIR="$SYSTEM_DATA"
export SCORECARD_SYSTEM_OUTPUT_DIR="$OUTPUT_DIR"

echo "SYSTEM_BASE=$SYSTEM_BASE"
echo "SCORECARD_CONFIG=$SCORECARD_CONFIG"
echo "SYSTEM_DATA=$SYSTEM_DATA"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RUNTIME_CONFIG_DIR=$RUNTIME_CONFIG_DIR"

cd "$SCRIPT_DIR"

if [ "$SCORECARD_RUN_CONFIG_SCORECARDS" = "true" ]; then
  echo ""
  echo "Running config-driven scorecard system..."
  python -m py_compile run_scorecard.py
  python run_scorecard.py --config-dir "$RUNTIME_CONFIG_DIR" --comparison all | tee "$LOG_DIR/run_config_driven_scorecards.log"
fi


if [ "$SCORECARD_RUN_SCORECARD_STYLE_RMSE" = "true" ]; then
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
fi

if [ "$SCORECARD_RUN_GLOBAL_4MODEL_PANELS" = "true" ]; then
  echo ""
  echo "Running global 4-model panel scripts..."
  for s in \
    plots_global_4models_check_aigfs_surface_humidity_vars.py \
    plots_global_4models_plot_global_4models_rmse_bias_panels.py
  do
    [ -f "$s" ] || continue

    echo ""
    echo "Compiling $s"
    python -m py_compile "$s"

    echo "Running $s"
    python "$s" | tee "$LOG_DIR/${s%.py}.log"
  done
fi

if [ "$SCORECARD_RUN_NESTED_0_48H" = "true" ]; then
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
fi

if [ "$SCORECARD_RUN_NESTED_D1_D10" = "true" ]; then
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
fi

if [ "$SCORECARD_RUN_GFS_DIFFERENCE" = "true" ]; then
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
fi

if [ "$SCORECARD_RUN_VIOLIN" = "true" ]; then
  echo ""
  echo "Running violin plot scripts..."
  for entry in \
    "plot_violin_all_responses_no_hrrr.py --config-dir $RUNTIME_CONFIG_DIR --plot-config $RUNTIME_CONFIG_DIR/all_response_violin_no_hrrr.yaml" \
    "plot_violin_all_responses_conus_hrrr_48h.py --config-dir $RUNTIME_CONFIG_DIR --plot-config $RUNTIME_CONFIG_DIR/all_response_violin_conus_hrrr_48h.yaml"
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
fi

echo ""
echo "Finished."
echo "Outputs:"
ls -lh "$OUTPUT_DIR"
