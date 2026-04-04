# TRELLIS UDF Decoder

## Project Overview

I'm building a **UDF (Unsigned Distance Field) decoder** within the [TRELLIS](https://github.com/microsoft/TRELLIS) 3D generation framework. The UDF decoder predicts distance from each grid vertex to the nearest **B-Rep (CAD) edge**, supervised by pre-computed geodesic distances. This is a new decoder head that runs in parallel to the existing mesh/Gaussian/radiance-field decoders — the encoder stays frozen, and we train only the UDF decoder.

I have been running this on an RTX 3080 with 10gb of VRAM, but you are now working on a Macbook M5 with 32gb unified memory. 

---

## TRELLIS Architecture Summary

TRELLIS uses a **Structured LATent (SLAT)** representation: sparse 3D voxel grid (64³, ~20K active voxels, 8 latent channels each). An encoder maps DINOv2 features to SLAT; multiple decoders convert SLAT to different representations.

Key architectural points:
- **Encoder**: 12-layer sparse transformer (768d, 12 heads, 3D Swin Attention window=8), 85.8M params. Frozen for decoder-only training.
- **Mesh Decoder (D_M)**: Same transformer backbone + 2 SparseSubdivideBlock3d upsamples (64³→128³→256³). Predicts FlexiCubes SDF + colors + normals at 8 corners per voxel. Supervised by multi-view rendered images (depth, normals, RGB). 90.9M params.
- **UDF Decoder**: Mirrors mesh decoder architecture but outputs only 8 channels (UDF at 8 cube corners) instead of FlexiCubes features. Uses `softplus` activation (output range (0,∞)). Supervised by pre-computed surface point UDF values. ~90.9M params.
- **Sparse Structure VAE**: Separate 3D Conv U-Net compressing binary occupancy 64³→16³ for generation stage.
- **Generation**: Two-stage rectified flow — (1) structure generation, (2) latent generation conditioned on text/image via CLIP/DINOv2 cross-attention.

For full details, see `trellis_reference_spec.md` in the repo. You will also find DATASET_STEP.md will also be helpful. 

---

Claude Code instructions:

This will be a long task and I expect you to delegate it out to subagents. 

Phase 1: Setup
    - Get NMR (see nmr/local-build.md) built. 
    - Get the step_renderer in trellis built. 
    - Download one chunk of the ABC dataset. 
        -https://archive.nyu.edu/bitstream/2451/44309/2/abc_0000_para_v00.7z
        -https://archive.nyu.edu/retrieve/89085/abc_0000_obj_v00.7z
        -https://archive.nyu.edu/bitstream/2451/44309/3/abc_0000_step_v00.7z
        you should only need the parasolid files, but I provided other links for reference. 

Phase 2: Replace GMSH with Parasolid
    - We're replacing GMSH with NMR/parasolid. You can see what we've already adjusted in NMR for processing ABC dataset in the most recent commit.
    - We need to finish making some changes to NMR to output what we need. See nmr/docs/superpowers/specs/2026-04-01-npz-edge-mesh.md
    - See parasolid_integration.md for what needs to be done in the trellis project. 

