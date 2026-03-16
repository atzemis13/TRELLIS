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
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'smooth_l1',
        lambda_reg: float = 0.1,
        lambda_edge_weight: float = 2.0,
        edge_threshold: float = 0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        self.lambda_edge_weight = lambda_edge_weight
        self.edge_threshold = edge_threshold

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
        batch_size: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Run inference on samples and produce UDF visualizations.

        Generates per-sample comparison images with:
          - GT UDF scatter (XY + XZ projections)
          - Predicted UDF scatter (same views)
          - Absolute error scatter (same views)
          - Summary metrics in the figure title

        Also generates a GT-vs-predicted UDF distribution histogram.
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        comparison_images = []
        all_gt_flat: List[np.ndarray] = []
        all_pred_flat: List[np.ndarray] = []

        samples_collected = 0
        for data in dataloader:
            if samples_collected >= num_samples:
                break

            args = recursive_to_device(data, 'cuda')
            reps = self.models['decoder'](args['latents'])
            pred_udf = self._sample_udf_at_points(reps, args['surface_points'])

            actual_batch = min(len(reps), num_samples - samples_collected)
            for j in range(actual_batch):
                if not reps[j].success:
                    continue

                pts = args['surface_points'][j].cpu().numpy()
                gt = args['surface_udf'][j].cpu().numpy()
                pred = pred_udf[j].cpu().numpy()

                comparison_images.append(
                    self._render_udf_comparison(pts, gt, pred)
                )
                all_gt_flat.append(gt)
                all_pred_flat.append(pred)

            samples_collected += actual_batch

        ret_dict: Dict = {}

        if comparison_images:
            ret_dict['udf_comparison'] = {
                'value': torch.stack(comparison_images),
                'type': 'image',
            }

        if all_gt_flat:
            gt_all = np.concatenate(all_gt_flat)
            pred_all = np.concatenate(all_pred_flat)
            ret_dict['udf_pred_vs_gt'] = {
                'value': self._render_pred_vs_gt_scatter(gt_all, pred_all).unsqueeze(0),
                'type': 'image',
            }

        return ret_dict

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _render_udf_comparison(
        self,
        points: np.ndarray,
        gt_values: np.ndarray,
        pred_values: np.ndarray,
    ) -> torch.Tensor:
        """
        Render a 3-row × 2-column comparison figure for one sample.

        Rows: GT UDF, Predicted UDF, Absolute Error.
        Columns: XY (top-down) and XZ (front) projections.

        Returns:
            [3, H, W] image tensor in [0, 1].
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        error = np.abs(pred_values - gt_values)

        # Adaptive colour ranges (95-th percentile, with sane minimums)
        udf_vmax = max(float(np.percentile(gt_values, 95)),
                       float(np.percentile(pred_values, 95)), 0.5)
        err_vmax = max(float(np.percentile(error, 95)), 0.05)

        fig, axes = plt.subplots(3, 2, figsize=(9, 12), dpi=100,
                                 layout='constrained')
        scatter_kw = dict(s=0.5, rasterized=True)

        rows = [
            ('GT UDF',    gt_values,   'viridis', 0, udf_vmax),
            ('Pred UDF',  pred_values, 'viridis', 0, udf_vmax),
            ('Error |Δ|', error,       'Reds',    0, err_vmax),
        ]

        for row_idx, (label, vals, cmap, vmin, vmax) in enumerate(rows):
            # XY projection — sort by Z for depth ordering
            order = np.argsort(points[:, 2])
            axes[row_idx, 0].scatter(
                points[order, 0], points[order, 1],
                c=vals[order], cmap=cmap, vmin=vmin, vmax=vmax, **scatter_kw)
            axes[row_idx, 0].set_xlim(-0.55, 0.55)
            axes[row_idx, 0].set_ylim(-0.55, 0.55)
            axes[row_idx, 0].set_aspect('equal')
            axes[row_idx, 0].set_title(f'{label} — XY')
            axes[row_idx, 0].set_xlabel('X')
            axes[row_idx, 0].set_ylabel('Y')

            # XZ projection — sort by Y for depth ordering
            order = np.argsort(points[:, 1])
            sc = axes[row_idx, 1].scatter(
                points[order, 0], points[order, 2],
                c=vals[order], cmap=cmap, vmin=vmin, vmax=vmax, **scatter_kw)
            axes[row_idx, 1].set_xlim(-0.55, 0.55)
            axes[row_idx, 1].set_ylim(-0.55, 0.55)
            axes[row_idx, 1].set_aspect('equal')
            axes[row_idx, 1].set_title(f'{label} — XZ')
            axes[row_idx, 1].set_xlabel('X')
            axes[row_idx, 1].set_ylabel('Z')

            fig.colorbar(sc, ax=axes[row_idx, :].tolist(), shrink=0.8, pad=0.02)

        # Summary metrics in the super-title
        mae = float(error.mean())
        rmse = float(np.sqrt((error ** 2).mean()))
        acc_005 = float((error < 0.05).mean() * 100)
        acc_01 = float((error < 0.1).mean() * 100)
        fig.suptitle(
            f'MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  |  '
            f'Acc@0.05: {acc_005:.1f}%  |  Acc@0.1: {acc_01:.1f}%',
            fontsize=11)

        return self._fig_to_tensor(fig)

    def _render_pred_vs_gt_scatter(
        self,
        gt_values: np.ndarray,
        pred_values: np.ndarray,
    ) -> torch.Tensor:
        """
        Render pred-vs-GT scatter plots (full range + zoomed near-edge).

        Left:  full range with identity line and metrics.
        Right: zoomed to GT < 2.0 (near-edge region).

        Returns:
            [3, H, W] image tensor in [0, 1].
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        error = np.abs(pred_values - gt_values)
        max_val = max(float(gt_values.max()), float(pred_values.max())) * 1.05
        max_val = max(max_val, 0.5)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # --- Left: full range ---
        ax = axes[0]
        ax.scatter(gt_values, pred_values, s=1, alpha=0.3, c='steelblue', rasterized=True)
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect')
        ax.set_xlabel('GT UDF')
        ax.set_ylabel('Pred UDF')
        ax.set_title('Pred vs GT (all points)')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        mae = float(error.mean())
        rmse = float(np.sqrt((error ** 2).mean()))
        acc_01 = float((error < 0.1).mean())
        acc_005 = float((error < 0.05).mean())
        ax.text(0.05, 0.95,
                f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nacc@0.1: {acc_01:.1%}\nacc@0.05: {acc_005:.1%}',
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # --- Right: near-edge zoom (GT < 2.0) ---
        ax = axes[1]
        mask = gt_values < 2.0
        if mask.sum() > 0:
            ax.scatter(gt_values[mask], pred_values[mask], s=2, alpha=0.3,
                       c='steelblue', rasterized=True)
            ax.plot([0, 2.0], [0, 2.0], 'r--', linewidth=1, label='Perfect')
            ne_err = error[mask]
            ne_mae = float(ne_err.mean())
            ne_acc = float((ne_err < 0.1).mean())
            ax.text(0.05, 0.95,
                    f'Near-edge MAE: {ne_mae:.4f}\nNear-edge acc@0.1: {ne_acc:.1%}',
                    transform=ax.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('GT UDF')
        ax.set_ylabel('Pred UDF')
        ax.set_title('Near-Edge Region (GT < 2.0)')
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 2.0)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)

        fig.tight_layout()
        return self._fig_to_tensor(fig)

    @staticmethod
    def _fig_to_tensor(fig) -> torch.Tensor:
        """Convert a matplotlib figure to a [3, H, W] float tensor in [0, 1]."""
        import matplotlib.pyplot as plt

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # buffer_rgba() works on all modern matplotlib versions
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)[:, :, :3].copy()  # drop alpha
        plt.close(fig)
        return torch.from_numpy(buf).float().div_(255.0).permute(2, 0, 1)
    
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
