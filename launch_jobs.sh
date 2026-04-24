#!/bin/bash
# Launch N parallel AWS Bedrock jobs, each processing a slice of MedPerturb.
#
# Usage:
#   ./launch_jobs.sh <num_jobs> [model_name]
#
# Examples:
#   ./launch_jobs.sh 5 llama70b3_3
#   ./launch_jobs.sh 10 gpt_oss_120b

set -e

NUM_JOBS=${1:-5}
MODEL=${2:-llama70b3_3}
LOG_DIR="logs/parallel"
mkdir -p "$LOG_DIR"

INTERVAL=$(python3 -c "print(1.0 / $NUM_JOBS)")
echo "[INFO] Launching $NUM_JOBS jobs for model=$MODEL (slice size=$INTERVAL)"

for ((i=0; i<NUM_JOBS; i++)); do
    START=$(python3 -c "print($i * $INTERVAL)")
    END=$(python3 -c "print(($i + 1) * $INTERVAL)")
    LOG="$LOG_DIR/job_${i}_${START}_${END}.log"
    echo "[INFO] Job $((i+1))/$NUM_JOBS: start=$START end=$END → $LOG"
    python run.py --mode AWS --name "$MODEL" --start "$START" --end "$END" \
        > "$LOG" 2>&1 &
done

wait
echo "[INFO] All $NUM_JOBS jobs completed."
