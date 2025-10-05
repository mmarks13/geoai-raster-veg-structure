# Evaluation

Inference, statistical analysis, and manuscript figure generation.

## Main Workflow

Scripts for generating published results and figures:

- `inference_eval.py` - Run model inference on test set
  - Loads trained model checkpoints
  - Generates predictions for test tiles
  - Computes Chamfer distance metrics
  - Saves results for downstream analysis

- `generate_eval_df.py` - Generate evaluation dataframes
  - Aggregates inference results into structured dataframes
  - Computes summary statistics
  - Prepares data for statistical tests

- `RQ_test_v2.py` - Statistical tests for research questions
  - Performs statistical hypothesis tests
  - Compares ablation study variants
  - Generates significance test results for manuscript

- `manuscript_figures.py` - Generate publication figures
  - Creates all figures used in the manuscript
  - Point cloud visualizations
  - Performance comparisons
  - Ablation study results

## Development Tools

Utilities for model development and analysis (not part of published workflow):

- `val_eval.py` - Validation evaluation utilities
  - Functions for evaluating models on validation set
  - Imported by `model_val_report.py`

- `model_val_report.py` - Generate validation reports
  - Creates detailed PDF reports for model validation
  - 3D point cloud visualizations
  - Per-sample metrics

- `model_comparison_report.py` - Multi-model comparison reports
  - Side-by-side model comparison with 3D visualizations
  - Chamfer distance comparisons
  - High-loss and high-improvement sample analysis

- `df_based_model_comparison_report.py` - DataFrame-based model comparison
  - Alternative comparison visualization approach
  - Uses pre-computed evaluation dataframes

## Typical Evaluation Workflow

1. **Run inference:**
   ```python
   python src/evaluation/inference_eval.py --model-path [checkpoint] --test-data [test_tiles.pt]
   ```

2. **Generate evaluation dataframes:**
   ```python
   python src/evaluation/generate_eval_df.py
   ```

3. **Perform statistical tests:**
   ```python
   python src/evaluation/RQ_test_v2.py
   ```

4. **Create manuscript figures:**
   ```python
   python src/evaluation/manuscript_figures.py
   ```

---

See [../../README.md](../../README.md) for complete workflow documentation.
