#!/bin/bash
# Idempotent, self-resuming launcher for the full-Laguna block run.
#
# Run this after every pod restart:   bash run_laguna.sh
# - A pgrep guard prevents launching a duplicate runner if one is already active.
# - An until-loop auto-resumes the runner if it exits non-zero (transient crash).
#   (Nothing user-space survives a full POD restart — just run this script again.)
# - The block runner itself is resumable: it skips the blocks already finished.
#
# Tuned for 8 CPUs: 3 blocks concurrent (one per GPU) x 3 H5 threads = 9 workers.

cd /home/jovyan/geoai-raster-veg-structure || exit 1

if pgrep -f "predict_polygon_blocks.py" >/dev/null; then
    echo "Runner already active (pid $(pgrep -f predict_polygon_blocks.py | tr '\n' ' '))."
    echo "Nothing to do. Tail progress with:  tail -f laguna_full.log"
    exit 0
fi

nohup bash -c '
  cd /home/jovyan/geoai-raster-veg-structure
  echo "[$(date)] launcher starting"
  until /home/jovyan/geoai_env/bin/python -u src/evaluation/predict_polygon_blocks.py \
        --polygon data/raw/inference_aoi/LagunaProjectArea.zip \
        --name Laguna_full --block-size 1000 --mc-passes 1 20 \
        --threads 3 --num-gpus 3; do
    echo "[$(date)] runner exited non-zero; resuming in 30s"; sleep 30
  done
  echo "[$(date)] runner finished (exit 0) — all blocks processed + mosaicked"
' >> laguna_full.log 2>&1 &

echo "Launched watchdog+runner (pid $!)."
echo "Progress:  tail -f laguna_full.log"
echo "Status:    python -c \"import json;d=json.load(open('data/output/polygon_predictions/Laguna_full/blocks_status.json'));import collections;print(collections.Counter(('ok' if v['status'] in ('ok','skipped (done)') else 'running' if v['status']=='running' else 'failed') for v in d['results'].values()))\""
