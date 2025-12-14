#!/usr/bin/env python3
"""
Test DPS with different scale values to find optimal guidance_scale
"""
import torch
import subprocess
import os
import yaml
import time

scale_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0]

print("=" * 70)
print("Testing DPS with different scale values")
print("=" * 70)
print(f"Will test {len(scale_values)} scale values: {scale_values}\n")

results = {}

for scale in scale_values:
    print(f"Testing scale={scale:6.3f}...", end=' ', flush=True)
    
    # Update config
    config_content = f"""name: DPS
method:
  _target_: algo.dps.DPS
  diffusion_scheduler_config:
    num_steps: 1000
    schedule: 'vp'
    timestep: 'vp'
    scaling: 'vp'
  guidance_scale: {scale}
  sde: True
"""
    
    with open('configs/algorithm/dps_toy.yaml', 'w') as f:
        f.write(config_content)
    
    # Run inference
    start_time = time.time()
    try:
        proc = subprocess.run(
            ['python', 'main.py', 'problem=toy_gausscmog8', 'algorithm=dps_toy', 'pretrain=toy_gausscmog8'],
            capture_output=True,
            text=True,
            timeout=200
        )
        
        # Check result file
        result_path = 'exps/inference/toy-gausscmog8/DPS/default/result_0.pt'
        if os.path.exists(result_path):
            data = torch.load(result_path)
            recon = data['recon']
            target = data['target']
            
            # Check for NaN/Inf
            if not torch.isfinite(recon).all():
                results[scale] = {'error': 'NaN/Inf in reconstruction', 'exploded': True}
                print("NaN/Inf detected")
                continue
            
            mse = ((recon - target) ** 2).mean().item()
            mae = (recon - target).abs().mean().item()
            recon_vec = recon.view(-1)
            max_abs = recon_vec.abs().max().item()
            
            # Check if exploded
            exploded = max_abs > 1e10 or mse > 1e10 or not torch.isfinite(torch.tensor([mse, mae])).all()
            
            results[scale] = {
                'mse': mse,
                'mae': mae,
                'max_abs': max_abs,
                'exploded': exploded,
                'time': time.time() - start_time
            }
            
            if exploded:
                print(f"EXPLODED - MSE={mse:.2e}, MaxAbs={max_abs:.2e}")
            else:
                print(f"OK - MSE={mse:.4f}, MAE={mae:.4f}, MaxAbs={max_abs:.2f}")
        else:
            print("No result file")
            results[scale] = {'error': 'No result file'}
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        results[scale] = {'error': 'Timeout'}
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        results[scale] = {'error': str(e)}

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Scale':>8s} {'Status':>12s} {'MSE':>15s} {'MAE':>12s} {'MaxAbs':>15s}")
print("-" * 70)

for scale in sorted(results.keys()):
    r = results[scale]
    if 'error' in r:
        print(f"{scale:8.3f} {'ERROR':>12s} {r['error']}")
    elif r.get('exploded', False):
        mse_str = f"{r['mse']:.2e}" if 'mse' in r else "N/A"
        max_str = f"{r['max_abs']:.2e}" if 'max_abs' in r else "N/A"
        print(f"{scale:8.3f} {'EXPLODED':>12s} {mse_str:>15s} {'N/A':>12s} {max_str:>15s}")
    else:
        print(f"{scale:8.3f} {'OK':>12s} {r['mse']:15.4f} {r['mae']:12.4f} {r['max_abs']:15.4f}")

# Find best scale
valid_results = {k: v for k, v in results.items() if 'error' not in v and not v.get('exploded', False)}
if valid_results:
    best_scale = min(valid_results.keys(), key=lambda k: valid_results[k]['mse'])
    best_mse = valid_results[best_scale]['mse']
    best_mae = valid_results[best_scale]['mae']
    
    print("\n" + "=" * 70)
    print(f"BEST SCALE: {best_scale}")
    print(f"  MSE: {best_mse:.4f}")
    print(f"  MAE: {best_mae:.4f}")
    print(f"  MaxAbs: {valid_results[best_scale]['max_abs']:.4f}")
    print("=" * 70)
    
    # Show top 3
    sorted_scales = sorted(valid_results.keys(), key=lambda k: valid_results[k]['mse'])
    print("\nTop 3 scales:")
    for i, scale in enumerate(sorted_scales[:3], 1):
        r = valid_results[scale]
        print(f"  {i}. scale={scale:6.3f}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")
else:
    print("\n" + "=" * 70)
    print("WARNING: No valid results found!")
    print("All scales either exploded or had errors.")
    print("=" * 70)





