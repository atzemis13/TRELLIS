"""
Dataset for Structured Latent to UDF training.

Extends the base SLAT dataset to include pre-computed surface points
with ground truth UDF values (distance to nearest BREP edge).
"""

import os
import json
import numpy as np
import torch
import utils3d.torch
from PIL import Image
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase


class SLat2UDF(StandardDatasetBase):
    """
    Dataset for Structured Latent and UDF supervision.
    
    Each sample includes:
    - SLAT latents (coords + feats)
    - Surface points uniformly sampled on the mesh
    - Ground truth UDF values at those points (distance to nearest BREP edge)
    
    Expected data format per instance:
    - latents/{latent_model}/{sha256}.npz: coords, feats
    - udf/{sha256}.npz: points [P, 3], udf [P]
    
    Args:
        roots: paths to the dataset (comma-separated)
        latent_model: latent model name
        num_surface_points: number of surface points to sample per instance
        min_aesthetic_score: minimum aesthetic score filter
        max_num_voxels: maximum number of voxels filter
        augment_points: whether to randomly subsample points each iteration
    """
    def __init__(
        self,
        roots: str,
        latent_model: str,
        num_surface_points: int = 10000,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        augment_points: bool = True,
    ):
        self.latent_model = latent_model
        self.num_surface_points = num_surface_points
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.augment_points = augment_points
        
        super().__init__(roots)
        
        # Required by trainer for normalization
        self.value_range = (0, 1)
        
    def filter_metadata(self, metadata):
        stats = {}
        
        # Filter for latent availability
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        
        # Filter for UDF availability
        if 'has_udf' in metadata.columns:
            metadata = metadata[metadata['has_udf']]
            stats['With UDF'] = len(metadata)
        
        # Aesthetic score filter
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        
        # Voxel count filter
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        
        return metadata, stats

    def _get_latent(self, root, instance):
        """Load SLAT latents."""
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        return {
            'coords': coords,
            'feats': feats,
        }
    
    def _get_udf(self, root, instance):
        """
        Load pre-computed surface points and UDF values.
        
        Expected npz format:
        - points: [P, 3] surface point positions in normalized coords [-0.5, 0.5]^3
        - udf: [P] unsigned distance to nearest BREP edge, pre-normalized by
          voxel size in the data prep pipeline (UDF=1.0 means one voxel width).
        """
        data = np.load(os.path.join(root, 'udf', f'{instance}.npz'))
        points = torch.tensor(data['points']).float()  # [P, 3]
        udf = torch.tensor(data['udf']).float()  # [P]
        
        # Subsample if we have more points than needed
        total_points = points.shape[0]
        if self.augment_points and total_points > self.num_surface_points:
            # Random subsample for data augmentation
            indices = torch.randperm(total_points)[:self.num_surface_points]
            points = points[indices]
            udf = udf[indices]
        elif total_points > self.num_surface_points:
            # Deterministic subsample (for evaluation)
            points = points[:self.num_surface_points]
            udf = udf[:self.num_surface_points]
        elif total_points < self.num_surface_points:
            # Pad with repeated points if we don't have enough
            # (shouldn't happen with proper data prep)
            repeat_factor = (self.num_surface_points // total_points) + 1
            points = points.repeat(repeat_factor, 1)[:self.num_surface_points]
            udf = udf.repeat(repeat_factor)[:self.num_surface_points]
        
        return {
            'surface_points': points,  # [P, 3]
            'surface_udf': udf,  # [P]
        }

    def get_instance(self, root, instance):
        latent = self._get_latent(root, instance)
        udf = self._get_udf(root, instance)
        return {
            **latent,
            **udf,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for sparse tensor batching.
        """
        pack = {}
        
        # Collate sparse latents (same as SLat2Render)
        coords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([
                torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                b['coords']
            ], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['latents'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        # Collate surface points and UDF: stack into [B, P, ...] tensors
        pack['surface_points'] = torch.stack([b['surface_points'] for b in batch])
        pack['surface_udf'] = torch.stack([b['surface_udf'] for b in batch])
        
        return pack
    
    def __str__(self):
        lines = [super().__str__()]
        lines.append(f'  - Num surface points: {self.num_surface_points}')
        lines.append(f'  - Augment points: {self.augment_points}')
        return '\n'.join(lines)


class SLat2RenderUDF(SLat2UDF):
    """
    Dataset combining rendered images with UDF supervision.
    
    Useful if you want to jointly train or visualize with rendered views.
    Extends SLat2UDF to also load rendered images.
    
    Args:
        roots: paths to the dataset
        image_size: rendered image size
        latent_model: latent model name
        num_surface_points: number of surface points per sample
        min_aesthetic_score: minimum aesthetic score
        max_num_voxels: maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        num_surface_points: int = 10000,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        **kwargs
    ):
        self.image_size = image_size
        super().__init__(
            roots=roots,
            latent_model=latent_model,
            num_surface_points=num_surface_points,
            min_aesthetic_score=min_aesthetic_score,
            max_num_voxels=max_num_voxels,
            **kwargs
        )
        
    def _get_image(self, root, instance):
        """Load a rendered image with camera parameters."""
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

    def get_instance(self, root, instance):
        latent = self._get_latent(root, instance)
        udf = self._get_udf(root, instance)
        image = self._get_image(root, instance)
        return {
            **latent,
            **udf,
            **image,
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function including image data."""
        pack = SLat2UDF.collate_fn(batch)
        
        # Add image data
        if 'image' in batch[0]:
            pack['image'] = torch.stack([b['image'] for b in batch])
            pack['alpha'] = torch.stack([b['alpha'] for b in batch])
            pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
            pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])
        
        return pack
