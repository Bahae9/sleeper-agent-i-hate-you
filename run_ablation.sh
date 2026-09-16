#!/usr/bin/env bash
# Runs the full ablation grid sequentially (single GPU). Stops on first
# failure so we don't burn GPU time on a cascading error.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=/home/bahae/venvs/ai-base/pytorch/bin/python

run() {
  local name=$1 n_total=$2 ratio=$3 epochs=$4
  echo "############ STARTING $name ############"
  mkdir -p "../results/$name"
  "$PY" run_pipeline.py --run-name "$name" --n-total "$n_total" --poison-ratio "$ratio" --epochs "$epochs" \
    2>&1 | tee "../results/$name/pipeline_log.txt"
  echo "############ DONE $name ############"
}

# baseline: 4,200 total, 10% poison, 1 epoch (re-run under the fixed eval/safety pools)
run baseline 4200 0.10 1

# isolate: does more training (same data) fix under-installation?
run more_epochs 4200 0.10 2

# isolate: does more absolute poison examples per position (same ratio) fix it?
run bigger_data 10000 0.10 1

# isolate: does higher poison density (same total size) fix it?
run higher_ratio 4200 0.20 1

echo "############ ALL EXPERIMENTS COMPLETE ############"
