'''
This file runs algorithms to solve inverse problems and evaluate the results.

Inference steps:
1. instantiate the forward model
2. instantiate the dataloder for test data
3. load the pretrained diffusion model
4. run the inference algorithm

Evaluation steps:
1. instantiate the evaluation metric(s)
2. evaluate the results
'''
import sys
import os
sys.path.append(os.getcwd())
import os
import training.dataset
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate


import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.helper import open_url, create_logger


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    # set random seed
    torch.manual_seed(config.seed)

    if config.wandb:
        problem_name = config.get('problem')['name']
        wandb.init(project=problem_name, group=config.algorithm.name, 
                   config=OmegaConf.to_container(config), 
                   reinit=True, settings=wandb.Settings(start_method="fork"))
        config = OmegaConf.create(dict(wandb.config)) # necessary for wandb sweep because wandb.config will be overwritten by sweep agent right after wandb.init
    # set up directory for logging and saving data
    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    # save config
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    forward_op = instantiate(config.problem.model, device=device)
    if config.problem.data._target_ == 'training.dataset.ToyDataset':
        data_config = OmegaConf.to_container(config.problem.data, resolve=True)
        data_config.pop('_target_')
        testset = instantiate(config.problem.data, problem=forward_op, **data_config)
    else:
        testset = instantiate(config.problem.data)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    logger.info(f"Loaded {len(testset)} test samples...")
    # load pre-trained model
    if config.pretrain.model._target_ == 'models.analytic.AnalyticPrior':
        net = instantiate(config.pretrain.model).to(device)
        logger.info("Instantiated analytic prior...")
    else:
        ckpt_path = config.problem.prior
        try:
            with open_url(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
                net = ckpt['ema'].to(device)
        except:
            net = instantiate(config.pretrain.model)
            ckpt = torch.load(config.problem.prior, map_location=device)
            # net.model.load_state_dict(ckpt)
            if 'ema' in ckpt.keys():
                net.load_state_dict(ckpt['ema'])
            else:
                net.load_state_dict(ckpt['net'])
            net = net.to(device)
        del ckpt
        logger.info(f"Loaded pre-trained model from {config.problem.prior}...")
    net.eval()
    if config.compile:
        net = torch.compile(net)
    # set up algorithm
    algo = instantiate(config.algorithm.method, forward_op=forward_op, net=net)
    # set up evaluator

    evaluator = instantiate(config.problem.evaluator, forward_op=forward_op)

    # Create images directory for toy problem visualization
    if config.problem.name in ['toy-gausscmog8', 'toy-image-lesion']:
        images_dir = os.path.join(exp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

    for i, data in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if config.inference:
            # get the observation
            observation = forward_op(data)
            target = data['target']
            # run the algorithm
            logger.info(f'Running inference on test sample {data_id}...')
            recon = algo.inference(observation, num_samples=config.num_samples)
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

            result_dict = {
                'observation': observation,
                'recon': forward_op.unnormalize(recon).cpu(),
                'target': forward_op.unnormalize(target).cpu(),
            }
            torch.save(result_dict, save_path)
            logger.info(f"Saved results to {save_path}.")
        else:
            # load the results
            result_dict = torch.load(save_path)
            logger.info(f"Loaded results from {save_path}.")

        # Save visualization images for toy problem (before evaluation)
        if config.problem.name == 'toy-gausscmog8':
            # Compute posterior variance for visualization
            posterior_result = forward_op.compute_posterior_variance(result_dict['observation'])
            save_visualization(result_dict, images_dir, data_id, forward_op, posterior_result=posterior_result)
            logger.info(f"Saved visualization images to {images_dir}/result_{data_id}.png")
        elif config.problem.name == 'toy-image-lesion':
            # Compute posterior variance for visualization
            posterior_result = forward_op.compute_posterior_variance(result_dict['observation'])
            save_visualization_image_lesion(result_dict, images_dir, data_id, forward_op, posterior_result=posterior_result)
            logger.info(f"Saved visualization images to {images_dir}/result_{data_id}.png")
        
        # Skip evaluation for toy problem if only visualization is needed
        # Uncomment the following lines if you want to skip MSE calculation:
        # if config.problem.name == 'toy-gausscmog8':
        #     continue
        
        # evaluate the results (may be skipped if skip_mse=True)
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=result_dict['observation'])
        if metric_dict:
            logger.info(f"Metric results: {metric_dict}...")
        else:
            logger.info("MSE calculation skipped. Only visualization saved.")

    # Skip final evaluation aggregation for toy problem if only visualization is needed
    if config.problem.name not in ['toy-gausscmog8', 'toy-image-lesion'] or config.get('evaluate', True):
        logger.info("Evaluation completed...")
        # aggregate the results
        metric_state = evaluator.compute()
        logger.info(f"Final metric results: {metric_state}...")
        if config.wandb:
            wandb.log(metric_state)
            wandb.finish()
    else:
        logger.info("Skipping evaluation aggregation. Check visualization images in the images/ directory.")
        if config.wandb:
            wandb.finish()


def save_visualization(result_dict, images_dir, data_id, forward_op, posterior_result=None):
    """
    Save visualization images for toy gausscmog8 problem.
    Creates side-by-side comparison of target, reconstruction, and observation.
    Also shows the 8D data mapping.
    """
    recon = result_dict['recon']  # [N, 1, 4, 4] or [1, 4, 4]
    target = result_dict['target']  # [1, 4, 4] or [N, 1, 4, 4]
    observation = result_dict['observation']  # [1, 4, 4] or [N, 1, 4, 4]
    
    # Handle different shapes
    if recon.dim() == 3:
        recon = recon.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if observation.dim() == 3:
        observation = observation.unsqueeze(0)
    
    # Take first sample if batch
    recon = recon[0, 0].detach().numpy()  # [4, 4]
    target = target[0, 0].detach().numpy()  # [4, 4]
    observation = observation[0, 0].detach().numpy()  # [4, 4]
    
    # Extract 8D data (first 8 positions)
    # For target and recon: always use first 8 dimensions (the true 8D data)
    target_8d = target.flatten()[:8]
    recon_8d = recon.flatten()[:8]
    
    # For observation: need to check A_obs_dim
    # If A_obs_dim < 8, observation has A_obs_dim real observations + padding
    # We should only show the real observation dimensions
    if hasattr(forward_op, 'A_obs_dim') and forward_op.A_obs_dim < 8:
        # Underdetermined case: only show first A_obs_dim dimensions
        obs_8d = observation.flatten()[:forward_op.A_obs_dim]
        # Pad with NaN or zeros for visualization
        obs_8d_padded = np.pad(obs_8d, (0, 8 - forward_op.A_obs_dim), constant_values=np.nan)
        obs_8d = obs_8d_padded
    else:
        # Determined or overdetermined: use first 8 dimensions
        obs_8d = observation.flatten()[:8]
    
    # Create figure with subplots
    # If posterior_result is provided, add an extra row for posterior variance visualization
    if posterior_result is not None:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.4, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3)
    
    # Find common scale for all images (based on 8D data only to avoid zero-padding affecting scale)
    target_8d = target.flatten()[:8]
    recon_8d = recon.flatten()[:8]
    obs_8d = observation.flatten()[:8]
    vmin = min(recon_8d.min(), target_8d.min(), obs_8d.min())
    vmax = max(recon_8d.max(), target_8d.max(), obs_8d.max())
    # Add some margin
    vrange = vmax - vmin if vmax > vmin else 1.0
    vmin = vmin - 0.1 * vrange
    vmax = vmax + 0.1 * vrange
    
    # Plot target
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Target (Ground Truth)\n[8D data in first 2 rows, zeros in last 2 rows]', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    # Add grid to show 8D mapping
    for i in range(5):
        ax1.axhline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
        ax1.axvline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
    # Highlight the 8D region
    rect = plt.Rectangle((-0.5, -0.5), 4, 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax1.add_patch(rect)
    ax1.text(1.5, -0.8, '8D data region', ha='center', color='red', fontsize=8)
    
    # Plot reconstruction
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Reconstruction', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    for i in range(5):
        ax2.axhline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
        ax2.axvline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
    rect2 = plt.Rectangle((-0.5, -0.5), 4, 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax2.add_patch(rect2)
    
    # Plot observation
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(observation, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title('Observation (y = A @ x0 + noise)', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3)
    for i in range(5):
        ax3.axhline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
        ax3.axvline(i-0.5, color='white', linewidth=0.5, alpha=0.3)
    rect3 = plt.Rectangle((-0.5, -0.5), 4, 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax3.add_patch(rect3)
    
    # Plot 8D values comparison
    ax4 = fig.add_subplot(gs[1, :])
    x_pos = np.arange(8)
    width = 0.25
    
    # Handle NaN values in obs_8d (for underdetermined case)
    obs_8d_valid = obs_8d.copy()
    obs_8d_valid_mask = ~np.isnan(obs_8d_valid)
    
    ax4.bar(x_pos - width, target_8d, width, label='Target 8D', alpha=0.8, color='blue')
    ax4.bar(x_pos, recon_8d, width, label='Reconstruction 8D', alpha=0.8, color='orange')
    # Only plot observation bars where we have real data
    if np.any(obs_8d_valid_mask):
        ax4.bar(x_pos[obs_8d_valid_mask] + width, obs_8d_valid[obs_8d_valid_mask], 
                width, label='Observation 8D', alpha=0.8, color='green')
        # Mark dimensions without observation data
        if not np.all(obs_8d_valid_mask):
            ax4.bar(x_pos[~obs_8d_valid_mask] + width, [0]*np.sum(~obs_8d_valid_mask), 
                    width, alpha=0.3, color='gray', label='No observation (padding)')
    else:
        ax4.bar(x_pos + width, obs_8d, width, label='Observation 8D', alpha=0.8, color='green')
    ax4.set_xlabel('Dimension (0-7)', fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.set_title('8D Data Comparison (first 8 dimensions only)', fontsize=10)
    #     ax4.legend()
    #     ax4.grid(True, alpha=0.3)
    #     ax4.axhline(0, color='black', linewidth=0.5)
    #     
    #     # Add posterior variance visualization if provided
    #     if posterior_result is not None:
    #         # Plot posterior variance as bar chart
    #         ax5 = fig.add_subplot(gs[2, :])
    #         
    #         # Use total mixture posterior variance if available, otherwise use common covariance diagonal
    #         if 'total_posterior_variance' in posterior_result:
    #             # Total variance includes both within-component and between-component variance
    #             posterior_var = posterior_result['total_posterior_variance'][0].detach().numpy()  # [8]
    #             var_label = 'Total Mixture Posterior Variance'
    #             title_suffix = ' (Total MoG Posterior)'
    #         else:
    #             # Common covariance diagonal (within-component variance only)
    #             posterior_var = posterior_result['posterior_variance_diag'].detach().numpy()  # [8]
    #             var_label = 'Common Posterior Variance (within-component)'
    #             title_suffix = ' (Common Covariance)'
    #         
    #         posterior_std = np.sqrt(posterior_var)
    #         
    #         x_pos = np.arange(8)
    #         width = 0.35
    #         ax5.bar(x_pos - width/2, posterior_var, width, label=var_label, alpha=0.8, color='purple')
    #         ax5_twin = ax5.twinx()
    #         ax5_twin.bar(x_pos + width/2, posterior_std, width, label='Posterior Std Dev', alpha=0.8, color='orange')
    #         
    #         ax5.set_xlabel('Dimension (0-7)', fontsize=10)
    #         ax5.set_ylabel('Variance', fontsize=10, color='purple')
    #         ax5_twin.set_ylabel('Standard Deviation', fontsize=10, color='orange')
    #         ax5.set_title(f'Posterior Variance and Standard Deviation (True Analytical{title_suffix})', fontsize=10)
    #         ax5.grid(True, alpha=0.3)
    #         ax5.axhline(0, color='black', linewidth=0.5)
    #         
    #         # Combine legends
    #         lines1, labels1 = ax5.get_legend_handles_labels()
    fig.tight_layout(pad=2.0)
    img_path = os.path.join(images_dir, f'result_{data_id}.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save error map
    error = np.abs(recon - target)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=error.max())
    ax.set_title('Absolute Error')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    error_path = os.path.join(images_dir, f'error_{data_id}.png')
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_visualization_image_lesion(result_dict, images_dir, data_id, forward_op, posterior_result=None):
    """
    Save visualization images for toy_image_lesion problem (16×16 images).
    Creates side-by-side comparison of target, reconstruction, and observation.
    Also shows posterior information if available.
    """
    recon = result_dict['recon']  # [N, 1, 16, 16] or [1, 16, 16]
    target = result_dict['target']  # [1, 16, 16] or [N, 1, 16, 16]
    observation = result_dict['observation']  # [1, 8, 8] or [N, 1, 8, 8]
    
    # Handle different shapes
    if recon.dim() == 3:
        recon = recon.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if observation.dim() == 3:
        observation = observation.unsqueeze(0)
    
    # Take first sample if batch
    recon = recon[0, 0].detach().cpu().numpy()  # [16, 16]
    target = target[0, 0].detach().cpu().numpy()  # [16, 16]
    observation = observation[0, 0].detach().cpu().numpy()  # [8, 8]
    
    # Create figure with subplots
    # If posterior_result is provided, add extra rows for posterior visualization
    if posterior_result is not None:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3)
    
    # Use separate scales for target/recon vs observation
    # Target and recon should be in similar range (if reconstruction is good)
    # But if recon is way off, use target's scale for target, and clip recon for visualization
    target_vmin, target_vmax = target.min(), target.max()
    target_range = target_vmax - target_vmin if target_vmax > target_vmin else 1.0
    target_vmin = target_vmin - 0.1 * target_range
    target_vmax = target_vmax + 0.1 * target_range
    
    # For reconstruction, use a reasonable scale (clip extreme values for visualization)
    # Use tighter bounds based on target range or prior scale
    if hasattr(forward_op, 'tau'):
        # Use 5*tau as reasonable visualization bound
        vis_max = max(2.0, 5.0 * forward_op.tau)
        recon_clipped = np.clip(recon, -vis_max, vis_max)
    else:
        recon_clipped = np.clip(recon, -2.0, 2.0)  # Default clip for visualization
    recon_vmin, recon_vmax = recon_clipped.min(), recon_clipped.max()
    recon_range = recon_vmax - recon_vmin if recon_vmax > recon_vmin else 1.0
    recon_vmin = recon_vmin - 0.1 * recon_range
    recon_vmax = recon_vmax + 0.1 * recon_range
    
    # Plot target (16×16) - use target's own scale
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(target, cmap='viridis', vmin=target_vmin, vmax=target_vmax)
    ax1.set_title('Target (Ground Truth)\n16×16 image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot reconstruction (16×16) - use clipped version with its own scale
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(recon_clipped, cmap='viridis', vmin=recon_vmin, vmax=recon_vmax)
    # Add note if values were clipped
    vis_max = max(2.0, 5.0 * forward_op.tau) if hasattr(forward_op, 'tau') else 2.0
    if recon.max() > vis_max or recon.min() < -vis_max:
        ax2.text(0.5, -0.1, f'Note: Values clipped to [-{vis_max:.1f}, {vis_max:.1f}]\nActual range: [{recon.min():.1f}, {recon.max():.1f}]',
                transform=ax2.transAxes, ha='center', fontsize=8, color='red')
    ax2.set_title('Reconstruction', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot observation (8×8, upsampled to 16×16 for visualization)
    ax3 = fig.add_subplot(gs[0, 2])
    # Upsample observation to 16×16 for better visualization
    observation_upsampled = zoom(observation, (2, 2), order=1)
    obs_vmin, obs_vmax = observation.min(), observation.max()
    obs_range = obs_vmax - obs_vmin if obs_vmax > obs_vmin else 1.0
    obs_vmin = obs_vmin - 0.1 * obs_range
    obs_vmax = obs_vmax + 0.1 * obs_range
    im3 = ax3.imshow(observation_upsampled, cmap='viridis', vmin=obs_vmin, vmax=obs_vmax)
    ax3.set_title('Observation (y = A @ x0 + noise)\n8×8 downsampled, upsampled for display', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot error map (use clipped recon for error calculation to avoid extreme values)
    error = np.abs(recon_clipped - target)
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(error, cmap='hot', vmin=0, vmax=error.max())
    ax4.set_title('Absolute Error (using clipped recon)', fontsize=11)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    # Add note about actual error if recon was clipped
    vis_max = max(2.0, 5.0 * forward_op.tau) if hasattr(forward_op, 'tau') else 2.0
    if recon.max() > vis_max or recon.min() < -vis_max:
        actual_error = np.abs(recon - target)
        ax4.text(0.5, -0.1, f'Actual max error: {actual_error.max():.1f}',
                transform=ax4.transAxes, ha='center', fontsize=8, color='red')
    
    # Plot cross-section through center (to show lesion pattern)
    ax5 = fig.add_subplot(gs[1, 1:])
    center_row = target.shape[0] // 2
    x_pos = np.arange(target.shape[1])
    ax5.plot(x_pos, target[center_row, :], 'b-', label='Target', linewidth=2, alpha=0.8)
    ax5.plot(x_pos, recon[center_row, :], 'r--', label='Reconstruction', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Pixel Position', fontsize=10)
    ax5.set_ylabel('Intensity', fontsize=10)
    ax5.set_title(f'Cross-section at row {center_row} (center row)', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add posterior visualization if available
    if posterior_result is not None:
        # Plot posterior means for both components
        ax6 = fig.add_subplot(gs[2, :])
        if 'posterior_means' in posterior_result:
            mu_post_0 = posterior_result['posterior_means']['component_0'][0].detach().cpu().numpy()  # [256]
            mu_post_1 = posterior_result['posterior_means']['component_1'][0].detach().cpu().numpy()  # [256]
            mu_post_0_img = mu_post_0.reshape(16, 16)
            mu_post_1_img = mu_post_1.reshape(16, 16)
            
            # Show both posterior means side by side
            ax6a = fig.add_subplot(gs[2, 0])
            # Use target scale for posterior means
            im6a = ax6a.imshow(mu_post_0_img, cmap='viridis', vmin=target_vmin, vmax=target_vmax)
            ax6a.set_title('Posterior Mean (Normal)\nμ₀^post', fontsize=10)
            ax6a.axis('off')
            plt.colorbar(im6a, ax=ax6a, fraction=0.046, pad=0.04)
            
            ax6b = fig.add_subplot(gs[2, 1])
            im6b = ax6b.imshow(mu_post_1_img, cmap='viridis', vmin=target_vmin, vmax=target_vmax)
            ax6b.set_title('Posterior Mean (Lesion)\nμ₁^post', fontsize=10)
            ax6b.axis('off')
            plt.colorbar(im6b, ax=ax6b, fraction=0.046, pad=0.04)
            
            # Show total posterior mean
            if 'total_posterior_mean' in posterior_result:
                mu_total = posterior_result['total_posterior_mean'][0].detach().cpu().numpy().reshape(16, 16)
                ax6c = fig.add_subplot(gs[2, 2])
                im6c = ax6c.imshow(mu_total, cmap='viridis', vmin=target_vmin, vmax=target_vmax)
                ax6c.set_title('Total Posterior Mean\nE[x|y]', fontsize=10)
                ax6c.axis('off')
                plt.colorbar(im6c, ax=ax6c, fraction=0.046, pad=0.04)
        
        # Plot posterior variance
        ax7 = fig.add_subplot(gs[3, :])
        if 'total_posterior_variance' in posterior_result:
            posterior_var = posterior_result['total_posterior_variance'][0].detach().cpu().numpy()  # [256]
            posterior_var_img = posterior_var.reshape(16, 16)
            posterior_std_img = np.sqrt(posterior_var_img)
            
            # Show variance and std side by side
            ax7a = fig.add_subplot(gs[3, 0])
            im7a = ax7a.imshow(posterior_var_img, cmap='plasma')
            ax7a.set_title('Posterior Variance\nVar[x|y]', fontsize=10)
            ax7a.axis('off')
            plt.colorbar(im7a, ax=ax7a, fraction=0.046, pad=0.04)
            
            ax7b = fig.add_subplot(gs[3, 1])
            im7b = ax7b.imshow(posterior_std_img, cmap='plasma')
            ax7b.set_title('Posterior Std Dev\n√Var[x|y]', fontsize=10)
            ax7b.axis('off')
            plt.colorbar(im7b, ax=ax7b, fraction=0.046, pad=0.04)
            
            # Show updated weights
            if 'updated_component_weights' in posterior_result:
                w_tilde = posterior_result['updated_component_weights'][0].detach().cpu().numpy()
                ax7c = fig.add_subplot(gs[3, 2])
                ax7c.axis('off')
                weights_text = f'Updated Component Weights:\n\n'
                weights_text += f'Normal (w₀): {w_tilde[0]:.4f}\n'
                weights_text += f'Lesion (w₁): {w_tilde[1]:.4f}\n\n'
                weights_text += f'Original Weights:\n'
                if 'original_component_weights' in posterior_result:
                    w_orig = posterior_result['original_component_weights'][0].detach().cpu().numpy()
                    weights_text += f'Normal (π₀): {w_orig[0]:.4f}\n'
                    weights_text += f'Lesion (π₁): {w_orig[1]:.4f}'
                ax7c.text(0.5, 0.5, weights_text, transform=ax7c.transAxes,
                         fontsize=11, verticalalignment='center', horizontalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout(pad=2.0)
    img_path = os.path.join(images_dir, f'result_{data_id}.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save error map separately
    error = np.abs(recon - target)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=error.max())
    ax.set_title('Absolute Error Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    error_path = os.path.join(images_dir, f'error_{data_id}.png')
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()