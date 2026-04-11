#!/usr/bin/env bash
# reproduce_benchmarks.sh — Reproduce all thesis benchmark results from scratch.
#
# Prerequisites:
#   - Python 3.12+ with the backend package installed (pip install -e .[dev])
#   - data/training_data.jsonl (raw labeled dataset from build_reviewed_execution_dataset)
#   - GPU recommended for transformer models but CPU works (slower)
#
# This script runs the full pipeline:
#   1. Clean the training data (remove placeholder prices, repair position states)
#   2. Benchmark all 5 models with 5-fold cross-validation on cleaned data
#   3. Extract feature importance and run statistical significance tests
#
# All outputs go to data/ directory.
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "Step 1: Data Cleanup"
echo "============================================================"
python -m app.services.interpretation.cleanup_training_data \
  --input data/training_data.jsonl \
  --output-clean data/training_data_clean.jsonl \
  --output-flagged data/training_data_flagged.jsonl \
  --report data/cleanup_report.json

echo ""
echo "============================================================"
echo "Step 2: 5-Fold Cross-Validation (all 5 models)"
echo "============================================================"
python -m app.services.interpretation.benchmark_models \
  --models logreg svm mlp distilbert modernbert \
  --cv 5 \
  --dataset data/training_data_clean.jsonl \
  --output data/benchmark_results.json \
  --device auto

echo ""
echo "============================================================"
echo "Step 3: Feature Importance + Statistical Tests"
echo "============================================================"
python -m app.services.interpretation.analyze_results \
  --dataset data/training_data_clean.jsonl \
  --benchmark-results data/benchmark_results.json \
  --output data/analysis_results.json \
  --top-k 15

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  data/cleanup_report.json         — cleanup statistics"
echo "  data/training_data_clean.jsonl    — cleaned dataset"
echo "  data/benchmark_results.json       — full CV results"
echo "  data/analysis_results.json        — feature importance + significance"
echo "============================================================"
