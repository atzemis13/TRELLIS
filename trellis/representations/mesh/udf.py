"""
UDF (Unsigned Distance Field) representation for BREP edge distance.

This module provides representations and utilities for predicting UDF values
at FlexiCubes grid vertices, which can then be interpolated to mesh vertices.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import construct_dense_grid, sparse_cube2verts, get_dense_attrs


class UDFExtractResult:
    """
    Result of UDF extraction from sparse features.
    
    Stores UDF values on a dense vertex grid that aligns with FlexiCubes,
    allowing interpolation to mesh vertices.
    
    Attributes:
        udf_grid: [res+1, res+1, res+1] dense UDF values at grid vertices
        grid_vertices: [(res+1)^3, 3] vertex positions in normalized coords
        res: grid resolution
        success: whether extraction succeeded
        reg_loss: regularization loss (training only)
    """
    def __init__(
        self,
        udf_grid: torch.Tensor,
        grid_vertices: torch.Tensor,
        res: int = 256,
    ):
        self.udf_grid = udf_grid  # [res+1, res+1, res+1]
        self.grid_vertices = grid_vertices  # [(res+1)^3, 3]
        self.res = res
        self.success = udf_grid is not None and udf_grid.numel() > 0
        
        # Training only
        self.reg_loss = None
    
    def interpolate_to_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Interpolate UDF values to arbitrary 3D points using trilinear interpolation.
        
        Args:
            points: [N, 3] points in normalized coordinates [-0.5, 0.5]^3
            
        Returns:
            [N] UDF values at the query points
        """
        return interpolate_udf_to_points(self.udf_grid, points, self.res)


class SparseFeatures2UDF:
    """
    Convert sparse voxel features to a dense UDF grid.
    
    This mirrors SparseFeatures2Mesh but only predicts UDF values at the
    8 corners of each voxel, then assembles them into a dense grid.
    
    Args:
        device: torch device
        res: grid resolution (default 256 to match mesh decoder output)
        normalize_udf: if True, normalize UDF by voxel size for scale invariance
    """
    def __init__(
        self, 
        device: str = "cuda", 
        res: int = 256,
        normalize_udf: bool = True,
    ):
        super().__init__()
        self.device = device
        self.res = res
        self.normalize_udf = normalize_udf
        
        # Voxel size for normalization (grid spans [-0.5, 0.5])
        self.voxel_size = 1.0 / res
        
        # Construct the dense vertex grid (same as FlexiCubes)
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_v = verts.to(self.device)  # [(res+1)^3, 3]
        self.reg_c = cube.to(self.device)   # [res^3, 8]
        
        self._calc_layout()
    
    def _calc_layout(self):
        """Define the feature layout for UDF prediction."""
        LAYOUTS = {
            'udf': {'shape': (8, 1), 'size': 8},  # UDF at 8 cube corners
        }
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start  # Total: 8
    
    def get_layout(self, feats: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """Extract a specific attribute from the feature tensor."""
        if name not in self.layouts:
            return None
        layout = self.layouts[name]
        return feats[:, layout['range'][0]:layout['range'][1]].reshape(-1, *layout['shape'])
    
    def __call__(
        self, 
        cubefeats: SparseTensor, 
        training: bool = False
    ) -> UDFExtractResult:
        """
        Convert sparse features to dense UDF grid.
        
        Args:
            cubefeats: SparseTensor with coords [N, 4] and feats [N, C]
                       coords[:, 0] is batch index, coords[:, 1:4] is spatial position
            training: whether in training mode (enables regularization)
            
        Returns:
            UDFExtractResult containing the dense UDF grid
        """
        coords = cubefeats.coords[:, 1:]  # [N, 3] spatial coords
        feats = cubefeats.feats  # [N, C]
        
        # Extract UDF predictions at 8 corners per voxel
        udf = self.get_layout(feats, 'udf')  # [N, 8, 1]
        
        # UDF should be non-negative (unsigned distance)
        # Use softplus for smooth, non-negative output
        udf = F.softplus(udf)
        
        # Optionally normalize by voxel size
        if self.normalize_udf:
            # UDF is in units of voxel size, making it scale-invariant
            # At inference, multiply by voxel_size to get actual distance
            pass  # Keep as-is, normalization happens at data prep
        
        # Convert sparse voxel UDF to dense vertex grid
        # sparse_cube2verts aggregates values from voxels sharing each vertex
        # cubes_to_verts expects [N, 8, M] (3D), so keep the last dim
        v_pos, v_udf, reg_loss = sparse_cube2verts(
            coords, 
            udf,  # [N, 8, 1]
            training=training
        )
        
        # Get dense UDF grid at vertices
        # v_pos: [M, 3] unique vertex positions
        # v_udf: [M, 1] UDF values at those vertices
        udf_dense = get_dense_attrs(
            v_pos, 
            v_udf,  # [M, 1] — already has channel dim from sparse_cube2verts
            res=self.res + 1,
            sdf_init=False  # Don't initialize with SDF bias
        )  # [(res+1)^3, 1] flattened
        
        udf_grid = udf_dense.squeeze(-1)  # [(res+1)^3]
        
        # Reshape to 3D grid for spatial operations
        res_v = self.res + 1
        udf_grid = udf_grid.view(res_v, res_v, res_v)
        
        # For vertices not covered by any voxel, set UDF to max value
        # (they are far from the surface)
        udf_grid = torch.where(
            udf_grid == 0,
            torch.ones_like(udf_grid) * (self.res * self.voxel_size),  # Max possible UDF
            udf_grid
        )
        
        result = UDFExtractResult(
            udf_grid=udf_grid,
            grid_vertices=self.reg_v,
            res=self.res,
        )
        
        if training:
            # Smoothness regularization: penalize large UDF gradients
            # This encourages smooth UDF fields
            grad_x = (udf_grid[1:, :, :] - udf_grid[:-1, :, :]).abs()
            grad_y = (udf_grid[:, 1:, :] - udf_grid[:, :-1, :]).abs()
            grad_z = (udf_grid[:, :, 1:] - udf_grid[:, :, :-1]).abs()
            smoothness_loss = (grad_x.mean() + grad_y.mean() + grad_z.mean()) / 3.0
            
            # Combine with vertex aggregation reg loss
            result.reg_loss = reg_loss + 0.01 * smoothness_loss
        
        return result


def interpolate_udf_to_points(
    udf_grid: torch.Tensor,
    points: torch.Tensor,
    res: int,
) -> torch.Tensor:
    """
    Interpolate UDF values from a dense grid to arbitrary 3D points.
    
    Uses trilinear interpolation via grid_sample.
    
    Args:
        udf_grid: [res+1, res+1, res+1] dense UDF values
        points: [N, 3] query points in normalized coordinates [-0.5, 0.5]^3
        res: grid resolution
        
    Returns:
        [N] interpolated UDF values
    """
    # grid_sample expects input in [N, C, D, H, W] format
    # and grid in [N, D_out, H_out, W_out, 3] with values in [-1, 1]
    
    # Convert points from [-0.5, 0.5] to [-1, 1] for grid_sample
    grid = points * 2.0  # [N, 3]
    
    # Reshape for grid_sample: [1, 1, 1, N, 3]
    grid = grid.view(1, 1, 1, -1, 3)
    
    # Reshape UDF grid: [1, 1, res+1, res+1, res+1]
    udf_vol = udf_grid.view(1, 1, res + 1, res + 1, res + 1)
    
    # Trilinear interpolation
    # Note: grid_sample uses (x, y, z) order but our grid is (z, y, x) indexed
    # We need to flip the grid coordinates
    grid_flipped = grid[..., [2, 1, 0]]  # Flip to match grid indexing
    
    sampled = F.grid_sample(
        udf_vol,
        grid_flipped,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )  # [1, 1, 1, 1, N]
    
    return sampled.view(-1)  # [N]


def interpolate_udf_to_mesh_vertices(
    udf_result: UDFExtractResult,
    mesh_vertices: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate UDF values to mesh vertices.
    
    This is the main utility for combining UDF decoder output with mesh decoder output.
    
    Args:
        udf_result: UDFExtractResult from UDF decoder
        mesh_vertices: [V, 3] mesh vertex positions in normalized coordinates
        
    Returns:
        [V] UDF values at mesh vertices
    """
    return udf_result.interpolate_to_points(mesh_vertices)


def compute_udf_on_edges(
    udf_result: UDFExtractResult,
    mesh_vertices: torch.Tensor,
    mesh_edges: torch.Tensor,
    num_samples_per_edge: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute UDF values along mesh edges.
    
    Useful for visualizing or analyzing edge proximity.
    
    Args:
        udf_result: UDFExtractResult from UDF decoder
        mesh_vertices: [V, 3] mesh vertex positions
        mesh_edges: [E, 2] edge vertex indices
        num_samples_per_edge: number of sample points per edge
        
    Returns:
        edge_udf_min: [E] minimum UDF along each edge
        edge_udf_mean: [E] mean UDF along each edge
    """
    # Sample points along each edge
    v0 = mesh_vertices[mesh_edges[:, 0]]  # [E, 3]
    v1 = mesh_vertices[mesh_edges[:, 1]]  # [E, 3]
    
    # Interpolation parameters
    t = torch.linspace(0, 1, num_samples_per_edge, device=mesh_vertices.device)
    t = t.view(1, -1, 1)  # [1, S, 1]
    
    # Sample points: [E, S, 3]
    edge_points = v0.unsqueeze(1) * (1 - t) + v1.unsqueeze(1) * t
    
    # Flatten and interpolate
    E, S, _ = edge_points.shape
    points_flat = edge_points.view(-1, 3)  # [E*S, 3]
    udf_flat = udf_result.interpolate_to_points(points_flat)  # [E*S]
    
    # Reshape and compute statistics
    udf_per_edge = udf_flat.view(E, S)  # [E, S]
    edge_udf_min = udf_per_edge.min(dim=1).values  # [E]
    edge_udf_mean = udf_per_edge.mean(dim=1)  # [E]
    
    return edge_udf_min, edge_udf_mean
