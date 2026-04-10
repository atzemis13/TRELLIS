# Parasolid Dataset Preparation

This document covers data preparation for **Parasolid (`.x_t`) CAD files** used to train the UDF decoder. It is an addendum to [DATASET.md](DATASET.md), which covers the standard TRELLIS pipeline for mesh/image datasets.

## Overview

The UDF decoder predicts geodesic distance from each grid vertex to the nearest B-Rep (CAD) edge. Training data is generated from Parasolid (`.x_t`) files through a pipeline managed by `prep_parasolid_dataset.py`, followed by the standard TRELLIS rendering/feature pipeline.

## Prerequisites

**NMR:** Build `nmr-server` from `/path/to/nmr` — see `nmr/docs/local-build-linux.md`. The binary is auto-detected by `prep_parasolid_dataset.py`, or set `NMR_BIN` env var.

**Python:** Standard TRELLIS environment plus:

```bash
pip install potpourri3d trimesh
```

## Pipeline

### Step 1: Convert + Compute UDF

`prep_parasolid_dataset.py` runs two tracked steps in one invocation:

1. **Convert** — Calls NMR with `--tess-max-width 0.02` to tessellate each `.x_t` file into a triangle mesh with B-Rep edge vertices identified. Outputs `converted/{sha256}.npz`. Tracked as `has_converted` in metadata.

2. **UDF** — Normalizes mesh to [-0.5, 0.5]³, computes geodesic UDF via heat method, samples surface points with edge-biased distribution. Outputs `meshes/{sha256}.obj` and `udf/{sha256}.npz`. Tracked as `has_mesh` and `has_udf` in metadata.

```bash
cd dataset_toolkits
python prep_parasolid_dataset.py ParasolidFiles \
    --source_dir /path/to/parasolid_files \
    --output_dir datasets/ParasolidFiles
```

Both steps skip already-completed instances. Progress is saved to CSV files that `build_metadata.py` merges.

> **CRITICAL:** The `--tess_max_width` flag (default 0.02) controls tessellation density. Without fine tessellation, NMR's coarse meshes have nearly all vertices on B-Rep edges, producing garbage UDF ground truth. Do not change this default without good reason.

### Step 2: Render Multiview Images

```bash
cd dataset_toolkits
python render_parasolid.py ParasolidFiles \
    --output_dir ../datasets/ParasolidFiles \
    --source_dir ../parasolid_files \
    --num_views 150 \
    --max_workers 4
```

This generates for each instance:
- `renders/{sha256}/000.png` … `149.png` — RGBA multiview images (1024x1024)
- `renders/{sha256}/transforms.json` — Camera matrices (NeRF-style)
- `renders/{sha256}/mesh.ply` — Mesh converted from OBJ (for voxelization)

### Step 3: Standard TRELLIS Pipeline (Voxelize → Features → Latents)

After rendering, run the remaining standard pipeline steps from [DATASET.md](DATASET.md):

```bash
cd dataset_toolkits

# Voxelize
python voxelize.py ParasolidFiles --output_dir ../datasets/ParasolidFiles --source_dir ../parasolid_files

# Extract DINO features (GPU)
python extract_feature.py --output_dir ../datasets/ParasolidFiles

# Encode SLat latents (GPU)
python encode_latent.py --output_dir ../datasets/ParasolidFiles

# Update metadata after each step
python build_metadata.py ParasolidFiles --output_dir ../datasets/ParasolidFiles --source_dir ../parasolid_files --from_file
```

### Step 4: Train UDF Decoder

```bash
cd /path/to/TRELLIS
python train.py \
    --config configs/vae/slat_vae_dec_udf_swin8_B_64l8_fp16.json \
    --output_dir results/udf_decoder \
    --data_dir datasets/ParasolidFiles
```

## prep_parasolid_dataset.py Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Directory to save processed data |
| `--tess_max_width` | float | 0.02 | Max facet width for NMR tessellation. **Do not increase.** |
| `--num_samples` | int | 100000 | Number of surface points to sample per instance |
| `--resolution` | int | 256 | Voxel grid resolution for UDF normalization |
| `--edge_bias` | float | 0.5 | Fraction of samples biased toward near-edge triangles |
| `--edge_threshold` | float | 2.0 | UDF threshold in voxel widths defining "near-edge" faces |
| `--instances` | str | None | Specific instances to process (comma-separated or file) |
| `--rank` | int | 0 | Shard index for distributed processing |
| `--world_size` | int | 1 | Total number of shards |
| `--max_workers` | int | 4 | Parallel workers for UDF computation |

### Output Format

Each instance produces `udf/{sha256}.npz` containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `points` | [P, 3] | float32 | Surface point positions in [-0.5, 0.5]³ |
| `udf` | [P] | float32 | UDF values (voxel-normalized, ≥ 0) |
| `scale` | [1] | float32 | Scale factor applied during normalization |
| `offset` | [3] | float32 | Centroid offset applied during normalization |
| `num_edges` | [1] | int32 | Number of B-Rep edge vertices found |

## Dataset Directory Structure

After full processing, `datasets/ParasolidFiles/` contains:

```
datasets/ParasolidFiles/
├── metadata.csv                     # Master index with per-instance boolean flags
├── raw/                             # Symlinked source .x_t files
├── converted/                       # NMR tessellated meshes with edge vertices (.npz)
├── meshes/                          # Normalized OBJ meshes in [-0.5, 0.5]³
├── udf/                             # Surface points + UDF ground truth (.npz)
├── renders/                         # Multiview rendered images + transforms.json
├── voxels/                          # Voxelized occupancy grids (.ply)
├── features/                        # DINOv2 sparse voxel features
└── latents/                         # Encoded SLat latents
    └── dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/
```

### metadata.csv Flags

| Flag | Set by | Meaning |
|------|--------|---------|
| `has_converted` | `prep_parasolid_dataset.py` | NMR conversion complete |
| `has_mesh` | `prep_parasolid_dataset.py` | Normalized OBJ mesh exists |
| `has_udf` | `prep_parasolid_dataset.py` | UDF ground truth computed |
| `rendered` | `render_parasolid.py` | Multiview renders complete |
| `voxelized` | `voxelize.py` | Voxel grid computed |
| `feature_{model}` | `extract_feature.py` | DINOv2 features extracted |
| `latent_{model}` | `encode_latent.py` | SLAT latents encoded |
