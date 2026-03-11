"""
Structured Latent VAE UDF Decoder.

This decoder predicts UDF (Unsigned Distance Field) values at FlexiCubes grid
vertices, representing distance to the nearest BREP edge. It mirrors the
architecture of SLatMeshDecoder for consistency.
"""

from typing import *
import torch
import torch.nn as nn
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations.mesh.udf import UDFExtractResult, SparseFeatures2UDF
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.
    
    This is identical to the one in mesh_dec.py but included here for
    self-containment. In practice, you may want to import from a shared location.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h


class SLatUDFDecoder(SparseTransformerBase):
    """
    Sparse Latent UDF Decoder.
    
    Predicts UDF values at FlexiCubes grid vertices from structured latents.
    Architecture mirrors SLatMeshDecoder with:
    - Sparse transformer backbone
    - Two upsampling blocks: 64³ → 128³ → 256³
    - Output layer predicting UDF at 8 corners per voxel
    
    Args:
        resolution: base resolution (64), output will be resolution*4 (256)
        model_channels: transformer hidden dimension
        latent_channels: input latent dimension
        num_blocks: number of transformer blocks
        num_heads: number of attention heads
        num_head_channels: channels per attention head (alternative to num_heads)
        mlp_ratio: MLP hidden dim ratio
        attn_mode: attention mode for sparse transformer
        window_size: window size for windowed attention
        pe_mode: positional encoding mode
        use_fp16: use FP16 for transformer
        use_checkpoint: use gradient checkpointing
        qk_rms_norm: use RMS norm for QK
        representation_config: config for UDF representation
    """
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config or {}
        
        # UDF extractor: converts sparse features to dense UDF grid
        # Output resolution is resolution * 4 = 256 (after two 2x upsamples)
        self.udf_extractor = SparseFeatures2UDF(
            res=self.resolution * 4,
        )
        self.out_channels = self.udf_extractor.feats_channels  # 8 (UDF at 8 corners)
        
        # Upsampling blocks: 64³ → 128³ → 256³
        # Same architecture as mesh decoder
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        
        # Output layer: predict UDF at 8 corners per voxel
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers for stable training start
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)
    
    def to_representation(self, x: sp.SparseTensor) -> List[UDFExtractResult]:
        """
        Convert a batch of network outputs to UDF representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of UDFExtractResult, one per batch element
        """
        ret = []
        for i in range(x.shape[0]):
            udf_result = self.udf_extractor(x[i], training=self.training)
            ret.append(udf_result)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[UDFExtractResult]:
        """
        Forward pass: structured latents → UDF representations.
        
        Args:
            x: SparseTensor with shape [batch, *, latent_channels]
            
        Returns:
            List of UDFExtractResult, one per batch element
        """
        # Sparse transformer backbone
        h = super().forward(x)
        
        # Upsample: 64³ → 128³ → 256³
        for block in self.upsample:
            h = block(h)
        
        # Ensure dtype consistency
        h = h.type(x.dtype)
        
        # Predict UDF at 8 corners per voxel
        h = self.out_layer(h)
        
        # Convert to UDF representations
        return self.to_representation(h)


class ElasticSLatUDFDecoder(SparseTransformerElasticMixin, SLatUDFDecoder):
    """
    SLAT VAE UDF decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
