#!/bin/bash
# Real-time monitoring script for training progress
# Usage: ./watch_training.sh [job_id]

JOB_ID="$1"

if [ -z "$JOB_ID" ]; then
    # Find the latest running job
    JOB_ID=$(squeue -u $USER -h -o "%i" | head -1)
    if [ -z "$JOB_ID" ]; then
        echo "No running jobs found. Please provide a job ID:"
        echo "  ./watch_training.sh <job_id>"
        exit 1
    fi
    echo "Monitoring latest job: $JOB_ID"
fi

LOG_FILE="outputs/train_toy_image_lesion_${JOB_ID}.log"

echo "Watching log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Watch the log file in real-time
if [ -f "$LOG_FILE" ]; then
    tail -f "$LOG_FILE"
else
    echo "Log file not found. Waiting for it to be created..."
    while [ ! -f "$LOG_FILE" ]; do
        sleep 2
    done
    tail -f "$LOG_FILE"
fi
