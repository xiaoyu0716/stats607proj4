# Demo Notebook - Quick Start Guide

## Overview

The `demo.ipynb` notebook provides a self-contained demonstration of the project's main functionality:

1. **Reconstruction with DPS and DAPS** on A=I (identity matrix) inverse problem
2. **Coverage Analysis** for uncertainty quantification

**Expected runtime**: ~20-30 minutes (depending on hardware)

## Prerequisites

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install the local package
pip install -e .
```

### 2. Required Files

Make sure the following files exist:
- `toy_gausscmog8_diffusion.pt` - Pretrained diffusion model (should be in project root)
- `configs/problem/toy_gausscmog8.yaml` - Problem configuration
- `configs/algorithm/dps_toy.yaml` - DPS algorithm configuration
- `configs/algorithm/daps_toy.yaml` - DAPS algorithm configuration

## Running the Demo

### Option 1: Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook demo.ipynb
```

Then run all cells sequentially.

### Option 2: JupyterLab

```bash
# Start JupyterLab
jupyter lab demo.ipynb
```

### Option 3: VS Code

Open `demo.ipynb` in VS Code and run cells using the interactive notebook interface.

## What the Demo Shows

### Part 1: Reconstruction (5-10 minutes)

- Loads the toy gausscmog8 problem with A=I (identity matrix)
- Generates a test sample with noisy observation
- Runs DPS and DAPS reconstruction algorithms
- Visualizes:
  - 4x4 image comparison (target, observation, DPS, DAPS)
  - 8D data comparison bar chart
  - Reconstruction error (MSE) comparison

**Expected Output:**
- `reconstruction_comparison.png` - Side-by-side 4x4 image comparison
- `reconstruction_8d_comparison.png` - 8D data bar chart
- Console output showing MSE errors

### Part 2: Coverage Analysis (10-20 minutes)

- Runs coverage analysis for DPS and DAPS
- Uses small parameters for demo (N=10, K=20)
- Evaluates whether 95% credible intervals contain true values
- Visualizes:
  - Per-dimension coverage bar charts
  - Global coverage statistics

**Expected Output:**
- `coverage_analysis.png` - Coverage bar charts for each method
- `coverage_results_summary.csv` - Detailed results table
- Console output showing coverage statistics

## Full Experiments

For full-scale experiments, modify the parameters:

### Reconstruction

Run from command line:
```bash
# DPS
python main.py problem=toy_gausscmog8 algorithm=dps_toy pretrain=toy_gausscmog8

# DAPS
python main.py problem=toy_gausscmog8 algorithm=daps_toy pretrain=toy_gausscmog8
```

Results will be saved to `exps/inference/toy-gausscmog8/` with visualization images.

### Coverage Analysis

Modify the notebook cell or run from command line:
```bash
python scripts/uq_simulation_analysis.py \
  --experiment coverage \
  --methods DPS DAPS \
  --N 200 \
  --K 100
```

## Troubleshooting

### Issue: Model file not found

**Error**: `FileNotFoundError: Model file not found: toy_gausscmog8_diffusion.pt`

**Solution**: Download the pretrained model from the project repository or train it using:
```bash
python train.py -cn toy_gausscmog8
```

### Issue: Module not found

**Error**: `ModuleNotFoundError: No module named 'algo'`

**Solution**: Make sure you've installed the package:
```bash
pip install -e .
```

### Issue: CUDA out of memory

**Solution**: The demo uses CPU by default. If you want to use GPU, modify the device setting in the notebook, but be aware of memory constraints.

### Issue: Coverage experiment takes too long

**Solution**: The demo uses small parameters (N=10, K=20). For faster testing, you can reduce further:
- N=5, K=10 (very fast, less accurate)
- N=10, K=20 (demo default, ~10-20 min)
- N=200, K=100 (full experiment, ~hours)

## Output Files

After running the demo, you'll have:

1. **Reconstruction outputs:**
   - `reconstruction_comparison.png`
   - `reconstruction_8d_comparison.png`

2. **Coverage outputs:**
   - `coverage_analysis.png`
   - `coverage_results_summary.csv`
   - `exps/demo_coverage/coverage_results_detailed.csv`
   - `exps/demo_coverage/coverage_DPS_detailed.csv`
   - `exps/demo_coverage/coverage_DAPS_detailed.csv`

## Notes

- The demo uses CPU by default for compatibility
- For GPU acceleration, change `device='cpu'` to `device='cuda'` in the notebook
- The coverage analysis uses smaller parameters (N=10, K=20) for demo purposes
- Full experiments should use N=200, K=100 for statistically significant results



