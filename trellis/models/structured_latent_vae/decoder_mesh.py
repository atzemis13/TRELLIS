from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import MeshExtractResult
from ...representations.mesh import SparseFeatures2Mesh
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

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
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} act_layers...', flush=True)
        h = self.act_layers(x)
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} sub(h)...', flush=True)
        h = self.sub(h)
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} sub(x)...', flush=True)
        x = self.sub(x)
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} out_layers (4 steps)...', flush=True)
        for i, layer in enumerate(self.out_layers):
            print(f'        [SparseSubdivideBlock3d] res={self.resolution}   out_layer {i+1}/4: {layer.__class__.__name__}...', flush=True)
            h = layer(h)
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} skip_connection...', flush=True)
        h = h + self.skip_connection(x)
        print(f'        [SparseSubdivideBlock3d] res={self.resolution} done.', flush=True)
        return h


class SLatMeshDecoder(SparseTransformerBase):
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
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
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
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
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
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            print(f'      [decoder] to_representation: extracting mesh {i+1}/{x.shape[0]}...', flush=True)
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        print(f'      [decoder] to_representation: done.', flush=True)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        print(f'      [decoder] forward: running transformer...', flush=True)
        h = super().forward(x)
        print(f'      [decoder] forward: running upsample blocks...', flush=True)
        for block_idx, block in enumerate(self.upsample):
            print(f'      [decoder] forward:   upsample block {block_idx+1}/{len(self.upsample)}...', flush=True)
            h = block(h)
        print(f'      [decoder] forward: running out_layer...', flush=True)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        print(f'      [decoder] forward: calling to_representation...', flush=True)
        return self.to_representation(h)
    

class ElasticSLatMeshDecoder(SparseTransformerElasticMixin, SLatMeshDecoder):
    """
    Slat VAE Mesh decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
