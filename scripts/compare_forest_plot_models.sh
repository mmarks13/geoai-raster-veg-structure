#!/bin/bash
#
# Cross-Model Comparison for Forest Plot Evaluations
#
# This script aggregates evaluation statistics from all models in
# data/output/forest_plot_evaluations/ and creates a summary CSV table
# for easy comparison.
#
# Usage:
#   bash scripts/compare_forest_plot_models.sh
#
# Output:
#   data/output/forest_plot_evaluations/model_comparison_summary.csv

set -e

EVAL_DIR="data/output/forest_plot_evaluations"
OUTPUT_CSV="$EVAL_DIR/model_comparison_summary.csv"

echo "========================================"
echo "CROSS-MODEL COMPARISON"
echo "========================================"
echo ""

# Check if evaluation directory exists
if [ ! -d "$EVAL_DIR" ]; then
    echo "Error: Evaluation directory not found: $EVAL_DIR"
    echo "Run evaluate_forest_plots.sh first to generate evaluations"
    exit 1
fi

# Find all comparison_stats.json files
STATS_FILES=$(find "$EVAL_DIR" -name "comparison_stats.json" -type f 2>/dev/null)

if [ -z "$STATS_FILES" ]; then
    echo "Error: No comparison_stats.json files found in $EVAL_DIR"
    echo "Run evaluate_forest_plots.sh first to generate evaluations"
    exit 1
fi

NUM_MODELS=$(echo "$STATS_FILES" | wc -l)
echo "Found $NUM_MODELS model evaluation(s)"
echo ""

# Create CSV header
echo "model_name,eval_date,TreeCover_n,TreeCover_R2,TreeCover_RMSE,TreeCover_MAE,TreeCover_bias,TreeCover_field_mean,TreeCover_pred_mean,TotalFuels_n,TotalFuels_R2,TotalFuels_RMSE,TotalFuels_MAE,TotalFuels_bias,TotalFuels_field_mean,TotalFuels_pred_mean" > "$OUTPUT_CSV"

# Process each stats file
for STATS_FILE in $STATS_FILES; do
    # Extract model name from path
    # Example: data/output/forest_plot_evaluations/raster_model_fused_20251203/comparison/comparison_stats.json
    MODEL_DIR=$(dirname "$(dirname "$STATS_FILE")")
    MODEL_NAME=$(basename "$MODEL_DIR")

    # Get modification date of stats file (eval date)
    EVAL_DATE=$(stat -c %y "$STATS_FILE" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$EVAL_DATE" ]; then
        # macOS fallback
        EVAL_DATE=$(stat -f %Sm -t %Y-%m-%d "$STATS_FILE" 2>/dev/null || echo "unknown")
    fi

    echo "Processing: $MODEL_NAME (evaluated $EVAL_DATE)"

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo "Warning: jq not found, using python fallback"

        # Python fallback to parse JSON
        python3 -c "
import json
import sys

with open('$STATS_FILE') as f:
    data = json.load(f)

# TreeCover stats
tc = data.get('TreeCover', {})
tc_n = tc.get('n', '')
tc_r2 = tc.get('r_squared', '')
tc_rmse = tc.get('rmse', '')
tc_mae = tc.get('mae', '')
tc_bias = tc.get('bias', '')
tc_field_mean = tc.get('field_mean', '')
tc_pred_mean = tc.get('pred_mean', '')

# TotalFuels stats
tf = data.get('TotalFuels', {})
tf_n = tf.get('n', '')
tf_r2 = tf.get('r_squared', '')
tf_rmse = tf.get('rmse', '')
tf_mae = tf.get('mae', '')
tf_bias = tf.get('bias', '')
tf_field_mean = tf.get('field_mean', '')
tf_pred_mean = tf.get('pred_mean', '')

print(f'$MODEL_NAME,$EVAL_DATE,{tc_n},{tc_r2},{tc_rmse},{tc_mae},{tc_bias},{tc_field_mean},{tc_pred_mean},{tf_n},{tf_r2},{tf_rmse},{tf_mae},{tf_bias},{tf_field_mean},{tf_pred_mean}')
" >> "$OUTPUT_CSV"
    else
        # Use jq to extract fields
        ROW="$MODEL_NAME,$EVAL_DATE"

        # TreeCover stats
        ROW="$ROW,$(jq -r '.TreeCover.n // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.r_squared // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.rmse // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.mae // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.bias // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.field_mean // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TreeCover.pred_mean // ""' "$STATS_FILE")"

        # TotalFuels stats
        ROW="$ROW,$(jq -r '.TotalFuels.n // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.r_squared // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.rmse // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.mae // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.bias // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.field_mean // ""' "$STATS_FILE")"
        ROW="$ROW,$(jq -r '.TotalFuels.pred_mean // ""' "$STATS_FILE")"

        echo "$ROW" >> "$OUTPUT_CSV"
    fi
done

echo ""
echo "========================================"
echo "COMPARISON COMPLETE"
echo "========================================"
echo ""
echo "Summary saved to: $OUTPUT_CSV"
echo ""

# Display summary table
if command -v column &> /dev/null; then
    echo "Summary table:"
    echo ""
    cat "$OUTPUT_CSV" | column -t -s ','
else
    echo "Summary table (install 'column' for better formatting):"
    echo ""
    cat "$OUTPUT_CSV"
fi

echo ""
echo "Key metrics:"
echo "  - TreeCover_R2: Coefficient of determination for canopy cover"
echo "  - TreeCover_RMSE: Root mean squared error (percent)"
echo "  - TotalFuels_R2: Coefficient of determination for fuel load"
echo "  - TotalFuels_RMSE: Root mean squared error (tons/acre)"
