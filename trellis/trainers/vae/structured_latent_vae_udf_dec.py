"""
Trainer for Structured Latent VAE UDF Decoder.

This trainer supervises UDF prediction by comparing predicted UDF values
at surface points against ground truth distances to BREP edges.
"""

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...representations.mesh.udf import UDFExtractResult, interpolate_udf_to_points
from ...modules.sparse import SparseTensor
from ...utils.data_utils import recursive_to_device


class SLatVaeUDFDecoderTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE UDF Decoder.
    
    Supervises UDF prediction using pre-computed surface points with
    ground truth UDF values (distance to nearest BREP edge).
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
        fp16_scale_growth (float): Scale growth for FP16.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type ('l1', 'smooth_l1', 'l2').
        lambda_reg (float): Regularization loss weight.
        lambda_edge_weight (float): Weight multiplier for points near edges.
        edge_threshold (float): UDF threshold for "near edge" classification.
        normalize_udf (bool): Whether UDF values are normalized by voxel size.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'smooth_l1',
        lambda_reg: float = 0.1,
        lambda_edge_weight: float = 2.0,
        edge_threshold: float = 0.05,
        normalize_udf: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        self.lambda_edge_weight = lambda_edge_weight
        self.edge_threshold = edge_threshold
        self.normalize_udf = normalize_udf

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """No-op: UDF dataset has no images to visualize."""
        pass
        
    def _compute_udf_loss(
        self,
        pred_udf: torch.Tensor,
        gt_udf: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute UDF prediction loss.
        
        Args:
            pred_udf: [N] predicted UDF values
            gt_udf: [N] ground truth UDF values
            weights: [N] optional per-point weights
            
        Returns:
            Scalar loss tensor
        """
        if self.loss_type == 'l1':
            loss = (pred_udf - gt_udf).abs()
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_udf, gt_udf, reduction='none', beta=0.01)
        elif self.loss_type == 'l2':
            loss = (pred_udf - gt_udf) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if weights is not None:
            loss = loss * weights
            return loss.sum() / (weights.sum() + 1e-8)
        else:
            return loss.mean()
    
    def _compute_edge_weights(self, gt_udf: torch.Tensor) -> torch.Tensor:
        """
        Compute per-point weights that emphasize points near edges.
        
        Points with small GT UDF (close to BREP edges) get higher weight.
        
        Args:
            gt_udf: [N] ground truth UDF values
            
        Returns:
            [N] weight values
        """
        # Points near edges (small UDF) get higher weight
        near_edge = gt_udf < self.edge_threshold
        weights = torch.ones_like(gt_udf)
        weights[near_edge] = self.lambda_edge_weight
        return weights
    
    def _sample_udf_at_points(
        self,
        reps: List[UDFExtractResult],
        surface_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample UDF values at surface points for each batch element.
        
        Args:
            reps: List of UDFExtractResult, one per batch element
            surface_points: [B, P, 3] surface point positions
            
        Returns:
            [B, P] predicted UDF values
        """
        batch_size = len(reps)
        num_points = surface_points.shape[1]
        
        pred_udf = torch.zeros(batch_size, num_points, device=surface_points.device)
        
        for i, rep in enumerate(reps):
            if rep.success:
                pred_udf[i] = interpolate_udf_to_points(
                    rep.udf_grid,
                    surface_points[i],
                    rep.res,
                )
        
        return pred_udf
    
    def training_losses(
        self,
        latents: SparseTensor,
        surface_points: torch.Tensor,
        surface_udf: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            latents: The [N x * x C] sparse latents
            surface_points: [B, P, 3] surface point positions in [-0.5, 0.5]^3
            surface_udf: [B, P] ground truth UDF values at surface points

        Returns:
            terms: dict with loss terms including "loss" key
            aux: dict with auxiliary information (empty for now)
        """
        # Forward pass through decoder
        reps = self.training_models['decoder'](latents)
        
        terms = edict(loss=0.0)
        
        # Regularization loss from UDF extractor
        reg_losses = [rep.reg_loss for rep in reps if rep.reg_loss is not None]
        if reg_losses:
            terms['reg_loss'] = sum(reg_losses) / len(reg_losses)
            terms['loss'] = terms['loss'] + terms['reg_loss'] * self.lambda_reg
        
        # Sample predicted UDF at surface points
        pred_udf = self._sample_udf_at_points(reps, surface_points)
        
        # Compute success mask (some extractions might fail)
        success_mask = torch.tensor([rep.success for rep in reps], device=latents.device)
        
        if success_mask.sum() > 0:
            # Filter to successful samples
            pred_udf_valid = pred_udf[success_mask]  # [B', P]
            gt_udf_valid = surface_udf[success_mask]  # [B', P]
            
            # Flatten for loss computation
            pred_flat = pred_udf_valid.flatten()  # [B' * P]
            gt_flat = gt_udf_valid.flatten()  # [B' * P]
            
            # Compute edge-aware weights if enabled
            if self.lambda_edge_weight > 1.0:
                weights = self._compute_edge_weights(gt_flat)
            else:
                weights = None
            
            # Main UDF loss
            terms['udf_loss'] = self._compute_udf_loss(pred_flat, gt_flat, weights)
            terms['loss'] = terms['loss'] + terms['udf_loss']
            
            # Additional metrics for logging
            with torch.no_grad():
                terms['udf_mae'] = (pred_flat - gt_flat).abs().mean()
                terms['udf_rmse'] = ((pred_flat - gt_flat) ** 2).mean().sqrt()
                
                # Accuracy at different thresholds
                for thresh in [0.01, 0.05, 0.1]:
                    acc = ((pred_flat - gt_flat).abs() < thresh).float().mean()
                    terms[f'udf_acc_{thresh}'] = acc
        
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run inference on a few samples for visualization.
        
        Returns:
            Dictionary with visualization data
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        
        ret_dict = {}
        
        # Collect samples
        all_pred_udf = []
        all_gt_udf = []
        all_surface_points = []
        
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = recursive_to_device(data, 'cuda')
            
            # Forward pass
            reps = self.models['decoder'](args['latents'])
            
            # Sample UDF at surface points
            pred_udf = self._sample_udf_at_points(reps, args['surface_points'])
            
            all_pred_udf.append(pred_udf[:batch])
            all_gt_udf.append(args['surface_udf'][:batch])
            all_surface_points.append(args['surface_points'][:batch])
        
        # Concatenate
        all_pred_udf = torch.cat(all_pred_udf, dim=0)  # [N, P]
        all_gt_udf = torch.cat(all_gt_udf, dim=0)  # [N, P]
        all_surface_points = torch.cat(all_surface_points, dim=0)  # [N, P, 3]
        
        # Per-sample UDF grid slices (middle slice of each sample)
        # Only return 'image' type entries — base snapshot() expects tensors
        # with .contiguous() and saves them via save_image.
        num_grid_vis = min(num_samples, len(all_pred_udf))
        grid_slices = []
        # Run decoder on last batch to get grids
        reps = self.models['decoder'](args['latents'])
        for i in range(min(num_grid_vis, len(reps))):
            if reps[i].success:
                udf_grid = reps[i].udf_grid
                mid_z = udf_grid.shape[2] // 2
                slice_img = udf_grid[:, :, mid_z].cpu()
                # Normalize to [0, 1] for visualization
                slice_img = slice_img / (slice_img.max() + 1e-8)
                grid_slices.append(slice_img)
        
        if grid_slices:
            grid_slices = torch.stack(grid_slices)  # [N, H, W]
            ret_dict['udf_grid_slices'] = {
                'value': grid_slices.unsqueeze(1).repeat(1, 3, 1, 1),  # [N, 3, H, W]
                'type': 'image',
            }
        
        return ret_dict
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: evaluation data loader
            num_batches: max number of batches to evaluate (None for all)
            
        Returns:
            Dictionary of metric names to values
        """
        self.models['decoder'].eval()
        
        total_mae = 0.0
        total_rmse_sq = 0.0
        total_points = 0
        acc_counts = {0.01: 0, 0.05: 0, 0.1: 0}
        
        for i, data in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
                
            args = recursive_to_device(data, 'cuda')
            reps = self.models['decoder'](args['latents'])
            pred_udf = self._sample_udf_at_points(reps, args['surface_points'])
            
            success_mask = torch.tensor([rep.success for rep in reps], device=pred_udf.device)
            if success_mask.sum() == 0:
                continue
            
            pred_flat = pred_udf[success_mask].flatten()
            gt_flat = args['surface_udf'][success_mask].flatten()
            
            n = len(pred_flat)
            total_mae += (pred_flat - gt_flat).abs().sum().item()
            total_rmse_sq += ((pred_flat - gt_flat) ** 2).sum().item()
            total_points += n
            
            for thresh in acc_counts:
                acc_counts[thresh] += ((pred_flat - gt_flat).abs() < thresh).sum().item()
        
        if total_points == 0:
            return {'mae': 0, 'rmse': 0}
        
        metrics = {
            'mae': total_mae / total_points,
            'rmse': (total_rmse_sq / total_points) ** 0.5,
        }
        for thresh in acc_counts:
            metrics[f'acc_{thresh}'] = acc_counts[thresh] / total_points
        
        return metrics
