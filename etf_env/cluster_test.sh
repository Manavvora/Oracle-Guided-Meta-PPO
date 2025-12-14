#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SERVER FULL-SCALE TEST SCRIPT
# -----------------------------------------------------------------------------
# This script runs the full pipeline at scale on the server:
# - Automatically detects number of components
# - Runs each stage with large NUM_TEST_RUNS (200k)
# -----------------------------------------------------------------------------

set -euo pipefail

echo "======================================================================="
echo "Running full-scale server test..."
echo "======================================================================="

# ---------------------
# 0. Configuration
# ---------------------
# Number of test episodes per component (200k)
NUM_TEST_RUNS=200000
# Inspection budget and cost settings
INITIAL_BUDGET=100.0
INSPECTION_COST=0.5
MAX_DRAWDOWN=0.20
# PPO training timesteps (200k)
TOTAL_PPO_TRAIN_STEPS=200000

# Base directories
RESULTS_DIR="results_server_full"
RF_DATA_DIR="data/rf_training_data"
RF_MODEL_PATH="${RESULTS_DIR}/models/rf_budget.pkl"
ORACLE_DIR="${RESULTS_DIR}/oracle_policies"
VANILLA_DIR="${RESULTS_DIR}/vanilla_models"
META_PPO_DIR="${RESULTS_DIR}/meta_ppo_models"
OUTPUT_DIR="${RESULTS_DIR}/test_outputs"
FINAL_PLOT="${RESULTS_DIR}/final_comparison_plot.png"

# Create directory structure
mkdir -p "${RF_DATA_DIR}" \
         "${ORACLE_DIR}" \
         "${VANILLA_DIR}" \
         "${META_PPO_DIR}" \
         "${OUTPUT_DIR}" \
         "$(dirname "${RF_MODEL_PATH}")"

# ---------------------
# 1. Feature Engineering & RF Training
# ---------------------
echo "\n--- Stage 1/5: Feature Engineering & Random Forest Training ---"
python3 generate_features.py --output_dir "${RF_DATA_DIR}"
python3 models/random_forest.py \
    --data_dir    "${RF_DATA_DIR}" \
    --output_model "${RF_MODEL_PATH}"
echo "✅ Stage 1 complete"

# ---------------------
# 2. Oracle Policy Generation
# ---------------------
echo "\n--- Stage 2/5: Oracle Policy Generation ---"
# Determine number of components by listing tickers or policies
# Assumes tickers.npy exists and shape = (n,)
N_COMPONENTS=$(python3 - << 'PYCODE'
import numpy as np
tickers = np.load('data/tickers.npy')
print(len(tickers))
PYCODE
)
echo "Detected $N_COMPONENTS components"
for i in $(seq 0 $((N_COMPONENTS-1))); do
  echo "→ Generating Oracle for component $i"
  python3 etf_oracle_policy.py \
    --asset_index $i \
    --output_dir "${ORACLE_DIR}/oracle_comp_${i}"
done
echo "✅ Stage 2 complete"

# ---------------------
# 3. PPO Training
# ---------------------
echo "\n--- Stage 3/5: Training PPO Models ---"
# Vanilla PPO
echo "→ Training Vanilla PPO"
python3 vanilla_meta_ppo_train.py \
  --n_components $N_COMPONENTS \
  --timesteps    $TOTAL_PPO_TRAIN_STEPS \
  --output_dir   "${VANILLA_DIR}"

# Oracle-Guided Meta-PPO
echo "→ Training Oracle-Guided Meta-PPO"
python3 oracle_guided_meta_ppo_train_refactored.py \
  --n_components $N_COMPONENTS \
  --feature_dir  "${RF_DATA_DIR}" \
  --budget_model "${RF_MODEL_PATH}" \
  --oracle_dir   "${ORACLE_DIR}" \
  --output_dir   "${META_PPO_DIR}" \
  --timesteps    $TOTAL_PPO_TRAIN_STEPS

echo "✅ Stage 3 complete"

# ---------------------
# 4. Model Evaluation
# ---------------------
echo "\n--- Stage 4/5: Evaluating Models (each $NUM_TEST_RUNS runs) ---"
# Baseline
python3 etf_realistic_baseline.py \
  --num_runs    $NUM_TEST_RUNS \
  --output_path "${OUTPUT_DIR}/baseline_survival_curve.npy"
# Vanilla PPO
python3 vanilla_meta_ppo_test.py \
  --ppo_dir    "${VANILLA_DIR}" \
  --num_runs   $NUM_TEST_RUNS \
  --output_path "${OUTPUT_DIR}/vanilla_survival_curve.npy"
# Oracle
python3 oracle_policy_test.py \
  --oracle_dir  "${ORACLE_DIR}" \
  --num_runs    $NUM_TEST_RUNS \
  --n_components $N_COMPONENTS \
  --initial_budget $INITIAL_BUDGET \
  --inspection_cost $INSPECTION_COST \
  --max_drawdown $MAX_DRAWDOWN \
  --output_path "${OUTPUT_DIR}/oracle_survival_curve.npy"
# Oracle-Guided PPO
python3 oracle_guided_meta_ppo_test_refactored.py \
  --ppo_dir      "${META_PPO_DIR}" \
  --oracle_dir   "${ORACLE_DIR}" \
  --num_runs     $NUM_TEST_RUNS \
  --n_components $N_COMPONENTS \
  --initial_budget $INITIAL_BUDGET \
  --inspection_cost $INSPECTION_COST \
  --max_drawdown $MAX_DRAWDOWN \
  --output_path  "${OUTPUT_DIR}/meta_ppo_survival_curve.npy"
echo "✅ Stage 4 complete"

# ---------------------
# 5. Plot Comparison
# ---------------------
echo "\n--- Stage 5/5: Plotting Comparison ---"
python3 plot_survival_curves.py \
  --baseline_results "${OUTPUT_DIR}/baseline_survival_curve.npy" \
  --vanilla_results  "${OUTPUT_DIR}/vanilla_survival_curve.npy" \
  --meta_ppo_results "${OUTPUT_DIR}/meta_ppo_survival_curve.npy" \
  --oracle_results   "${OUTPUT_DIR}/oracle_survival_curve.npy" \
  --output_path      "${FINAL_PLOT}"
echo "✅ Stage 5 complete"

echo "\n======================================================================="
echo "🎉 Full-scale server test COMPLETED. Results in '${RESULTS_DIR}'"
echo "======================================================================="
