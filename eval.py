'''
This file implements evaluator for each problem.
'''
import torch
import numpy as np
from abc import ABC, abstractmethod
import torch
import numpy as np
from piq import LPIPS, psnr, ssim
from collections import defaultdict
from training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss


class Evaluator(ABC):
    def __init__(self, 
                 metric_list, 
                 forward_op=None, 
                 data_misfit=False):
        self.metric_list = metric_list
        self.forward_op = forward_op
        self.data_misfit = data_misfit
        if data_misfit:
            assert forward_op is not None, "forward_op must be provided for data misfit evaluation"
        
        self.device = forward_op.device if forward_op is not None else 'cpu'
        self.metric_state = {key: [] for key in metric_list.keys()}
        if data_misfit:
            self.metric_state['data misfit'] = []
        # each metric is a list of values

    def eval_data_misfit(self, pred, observation):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W) unnormalized
            - observation (torch.Tensor): (N, C, H, W)
        Returns:
            - data_misfit (torch.Tensor): (N,), data misfit
        '''
        data_misfit = self.forward_op.loss(pred, observation, unnormalize=False)
        return torch.sqrt(data_misfit)

    @abstractmethod
    def __call__(self, pred, target, observation=None, forward_op=None):
        ''''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
            - observation (torch.Tensor): (N, *observation.shape) or (*observation.shape)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        pass

    def compute(self):
        '''
        Returns:
            - metric_state (Dict): a dictionary of metric values
        '''
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state


class BlackHoleEvaluator(Evaluator):
    def __init__(self, 
                 forward_op=None):
        metric_list = {'cp_chi2': None, 'camp_chi2': None, 'psnr': None, 'blur_psnr (f=10)': None,
                       'blur_psnr (f=15)': None,
                       'blur_psnr (f=20)': None}
        super().__init__(metric_list, forward_op=forward_op)
        self.device = forward_op.device

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        pred, target, observation = pred.to(self.device), target.to(self.device), observation.to(self.device)

        # evaluation
        if pred.shape != target.shape:
            target = target.repeat(pred.shape[0], 1, 1, 1)
            observation = observation.repeat(pred.shape[0], 1, 1, 1)

        # chi-square
        chisq_cp, chisq_logcamp = self.forward_op.evaluate_chisq(pred, observation, True)

        # blurry PSNR
        blur_factors = [0, 10, 15, 20]
        blur_psnr = self.forward_op.evaluate_psnr(target, pred, blur_factors)
        blur_psnr = blur_psnr.max(dim=0)[0]

        metric_dict['cp_chi2'] = chisq_cp.min().item()
        metric_dict['camp_chi2'] = chisq_logcamp.min().item()
        metric_dict['psnr'] = blur_psnr[0].item()
        metric_dict['blur_psnr (f=10)'] = blur_psnr[1].item()
        metric_dict['blur_psnr (f=15)'] = blur_psnr[2].item()
        metric_dict['blur_psnr (f=20)'] = blur_psnr[3].item()

        self.metric_state['cp_chi2'].append(metric_dict['cp_chi2'])
        self.metric_state['camp_chi2'].append(metric_dict['camp_chi2'])
        self.metric_state['psnr'].append(metric_dict['psnr'])
        self.metric_state['blur_psnr (f=10)'].append(metric_dict['blur_psnr (f=10)'])
        self.metric_state['blur_psnr (f=15)'].append(metric_dict['blur_psnr (f=15)'])
        self.metric_state['blur_psnr (f=20)'].append(metric_dict['blur_psnr (f=20)'])
        return metric_dict
    

def relative_l2(pred, target):
    ''''
    Args:
        - pred (torch.Tensor): (N, C, H, W)
        - target (torch.Tensor): (C, H, W)
    Returns:
        - rel_l2 (torch.Tensor): (N,), relative L2 error
    '''
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2


class NavierStokes2d(Evaluator):
    def __init__(self, forward_op):
        metric_list = {'relative l2': relative_l2}
        super(NavierStokes2d, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        return metric_dict


class Image(Evaluator):
    def __init__(self, forward_op=None, min_size_for_lpips=32):
        self.eval_batch = 32
        self.min_size_for_lpips = min_size_for_lpips
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none')}
        # LPIPS requires at least 32x32 images, so we'll add it conditionally
        # Store LPIPS model separately to add it only when needed
        self.lpips_model = LPIPS(replace_pooling=True, reduction='none')
        super(Image, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        # Check image size for LPIPS (LPIPS requires at least 32x32)
        img_size = min(pred.shape[-2], pred.shape[-1]) if len(pred.shape) >= 3 else 0
        use_lpips = img_size >= self.min_size_for_lpips
        
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if pred.shape != target.shape:
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name].append(metric_dict[metric_name])
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        
        # Add LPIPS only for large enough images
        if use_lpips:
            if pred.shape != target.shape:
                target = target.repeat(pred.shape[0], 1, 1, 1)
            try:
                lpips_val = self.lpips_model(pred, target).mean().item()
                metric_dict['lpips'] = lpips_val
                if 'lpips' not in self.metric_state:
                    self.metric_state['lpips'] = []
                self.metric_state['lpips'].append(lpips_val)
            except Exception as e:
                # Skip LPIPS if it fails (e.g., image too small)
                pass
        
        return metric_dict

def fwi_norm(x):
    return (x - 1.5) / 3.0


class AcousticWave(Evaluator):
    def __init__(self, forward_op=None):
        metric_list = {'relative l2': relative_l2, 
                       'psnr': lambda x, y: psnr(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none')}
        super(AcousticWave, self).__init__(metric_list, forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {'data misfit': 0.0}
        for metric_name, metric_func in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
                self.metric_state[metric_name].append(val)
        
        data_misfit = self.eval_data_misfit(pred, observation).mean().item()
        metric_dict['data misfit']= data_misfit
        self.metric_state['data misfit'].append(data_misfit)
        return metric_dict
    
    
class MRI(Evaluator):
    def __init__(self, forward_op=None):
        dr_psnr_loss = DynamicRangePSNRLoss()
        dr_ssim_loss = DynamicRangeSSIMLoss()
        self.eval_batch = 32
        metric_list = {
            'psnr': lambda x, y: -dr_psnr_loss(x, y),
            'ssim': lambda x, y: 1-dr_ssim_loss(x, y)
        }
        super(MRI, self).__init__(metric_list, forward_op=forward_op)
        self.metric_state = defaultdict(list)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if len(pred) != len(target):
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name].append(metric_dict[metric_name])
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        if self.forward_op is not None and observation is not None:
            pred = pred.to(self.device)
            observation = observation.to(self.device)
            metric_dict['data misfit'] = torch.linalg.norm(self.forward_op.forward(pred) - observation).item()
            self.metric_state['data misfit'].append(metric_dict['data misfit'])
        return metric_dict



class InverseScatter(Evaluator):
    def __init__(self, forward_op=None):
        self.eval_batch = 32
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none')}
        super(InverseScatter, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if pred.shape != target.shape:
                val = metric_func(pred, target.repeat(pred.shape[0],1,1,1)).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        return metric_dict


class ToyEvaluator(Evaluator):
    """
    Evaluator for the toy problem, calculating Mean Squared Error (MSE).
    Only evaluates on the first 8 dimensions (the real data, not the zero-padded dimensions).
    Set skip_mse=True to skip MSE calculation and only save visualizations.
    """
    def __init__(self, forward_op=None, skip_mse=False):
        if skip_mse:
            metric_list = {}  # Empty metric list to skip calculation
        else:
            metric_list = {'mse': lambda pred, target: torch.mean((pred - target) ** 2)}
        super(ToyEvaluator, self).__init__(metric_list, forward_op=forward_op)
        self.skip_mse = skip_mse

    def __call__(self, pred, target, observation=None):
        if self.skip_mse:
            # Skip MSE calculation, just return empty dict
            return {}
        
        metric_dict = {}
        
        # Reshape both pred and target to be 2D tensors (batch_size, dim)
        pred_reshaped = pred.view(pred.shape[0], -1)
        target_reshaped = target.view(target.shape[0], -1)

        # Ensure target is broadcastable to pred
        if pred_reshaped.shape[0] != target_reshaped.shape[0] and target_reshaped.shape[0] == 1:
            target_reshaped = target_reshaped.expand_as(pred_reshaped)

        # Only evaluate on the first 8 dimensions (the real data)
        # The last 8 dimensions are zero-padding
        dim_true = 8
        pred_8d = pred_reshaped[:, :dim_true]
        target_8d = target_reshaped[:, :dim_true]
        
        # Check for invalid values and clip if necessary
        if torch.isnan(pred_8d).any() or torch.isinf(pred_8d).any():
            # Replace inf/nan with large finite values for evaluation
            pred_8d = torch.where(torch.isnan(pred_8d) | torch.isinf(pred_8d), 
                                 torch.zeros_like(pred_8d), pred_8d)
        
        # Clip extreme values to prevent overflow in MSE calculation
        pred_8d = torch.clamp(pred_8d, min=-1e6, max=1e6)
        target_8d = torch.clamp(target_8d, min=-1e6, max=1e6)

        for metric_name, metric_func in self.metric_list.items():
            val = metric_func(pred_8d, target_8d).mean().item()
            # Check if result is valid
            if not (np.isfinite(val) and val >= 0):
                val = float('inf')
            metric_dict[metric_name] = val
            self.metric_state[metric_name].append(val)
            
        return metric_dict
