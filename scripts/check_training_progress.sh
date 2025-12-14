#!/bin/bash
# Script to check training progress in multiple ways

echo "=== 1. Current Job Queue (squeue) ==="
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"

echo ""
echo "=== 2. Recent Job History (sacct) ==="
sacct -u $USER --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,NodeList -S $(date -d '1 day ago' +%Y-%m-%d) | head -20

echo ""
echo "=== 3. Latest Training Log ==="
LATEST_LOG=$(find outputs -name "train_toy_image_lesion_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "Log file: $LATEST_LOG"
    echo "Last 20 lines:"
    tail -20 "$LATEST_LOG"
    echo ""
    echo "File size: $(du -h "$LATEST_LOG" | cut -f1)"
    echo "Last modified: $(stat -c %y "$LATEST_LOG" 2>/dev/null || stat -f %Sm "$LATEST_LOG" 2>/dev/null)"
else
    echo "No training log found"
fi

echo ""
echo "=== 4. Latest Error Log ==="
LATEST_ERR=$(find outputs -name "train_toy_image_lesion_*.err" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_ERR" ] && [ -f "$LATEST_ERR" ]; then
    echo "Error log: $LATEST_ERR"
    if [ -s "$LATEST_ERR" ]; then
        echo "Last 10 lines:"
        tail -10 "$LATEST_ERR"
    else
        echo "No errors (file is empty)"
    fi
else
    echo "No error log found"
fi

echo ""
echo "=== 5. Model File Status ==="
if [ -f "toy_image_lesion_diffusion.pt" ]; then
    echo "Model file exists:"
    ls -lh toy_image_lesion_diffusion.pt
    echo "Last modified: $(stat -c %y toy_image_lesion_diffusion.pt 2>/dev/null || stat -f %Sm toy_image_lesion_diffusion.pt 2>/dev/null)"
else
    echo "Model file not created yet"
fi

echo ""
echo "=== 6. Training Progress (from log) ==="
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    # Extract step and loss information
    grep -E "step [0-9]+ \| loss=" "$LATEST_LOG" | tail -5
    if [ $? -ne 0 ]; then
        echo "No training progress found in log yet"
    fi
fi

echo ""
echo "=== 7. GPU Usage (if job is running) ==="
RUNNING_JOB=$(squeue -u $USER -h -o "%i" | head -1)
if [ -n "$RUNNING_JOB" ]; then
    echo "Running job ID: $RUNNING_JOB"
    echo "To check GPU usage on the compute node, use:"
    echo "  srun --jobid=$RUNNING_JOB --pty bash -c 'nvidia-smi'"
else
    echo "No running jobs"
fi
