# STEP File Dataset Preparation

This document covers data preparation for **STEP (CAD) files** used to train the UDF decoder. It is an addendum to [DATASET.md](DATASET.md), which covers the standard TRELLIS pipeline for mesh/image datasets.

## Overview

The UDF decoder predicts geodesic distance from each grid vertex to the nearest B-Rep (CAD) edge. Training data is generated from STEP files through a dedicated pipeline that:

1. Meshes the STEP geometry (gmsh)
2. Extracts B-Rep edge topology
3. Computes geodesic distance fields (heat method)
4. Samples surface points with edge-biased distribution
5. Produces `.npz` files compatible with the `SLat2UDF` dataset class

After STEP-specific processing, the standard TRELLIS pipeline (render, voxelize, extract features, encode latents) is run as described in [DATASET.md](DATASET.md).

## Prerequisites

Standard TRELLIS environment plus:

```bash
pip install gmsh potpourri3d trimesh
```

## Pipeline

### Step 0: Prepare STEP Files

Place STEP files in a source directory (e.g., `step_files/`). Each file should have a `.step` or `.stp` extension.

### Step 1: Process STEP Files

```bash
cd dataset_toolkits
python prep_step_dataset.py STEPFiles \
    --source_dir /path/to/step_files \
    --output_dir datasets/STEPFiles
```

This generates for each STEP file:
- `meshes/{sha256}.obj` — Normalized triangulated mesh in [-0.5, 0.5]³
- `udf/{sha256}.npz` — Surface points with ground truth UDF values
- `metadata.csv` — Dataset metadata index

### Step 2: Render Multiview Images

STEP files cannot be imported by Blender, so we use `render_step.py` with the custom `step_renderer` binary instead of the standard `render.py`.

**Prerequisites**: Build the step_renderer first:
```bash
cd step_renderer && mkdir -p build && cd build && cmake .. && make
```

Then render:
```bash
cd dataset_toolkits
python render_step.py STEPFiles \
    --output_dir ../datasets/STEPFiles \
    --source_dir ../step_files \
    --num_views 150 \
    --max_workers 4
```

This generates for each instance:
- `renders/{name}/000.png` … `149.png` — RGBA multiview images (512×512)
- `renders/{name}/transforms.json` — Camera matrices (NeRF-style, needed by feature extraction)
- `renders/{name}/views.json` — Camera parameters (yaw, pitch, radius, fov)
- `renders/{name}/mesh.ply` — Mesh converted from OBJ (needed by voxelization)

### Step 3: Standard TRELLIS Pipeline (Voxelize → Features → Latents)

After rendering, run the remaining standard pipeline steps from [DATASET.md](DATASET.md):

```bash
cd dataset_toolkits

# Voxelize
python voxelize.py STEPFiles --output_dir ../datasets/STEPFiles --source_dir ../step_files

# Extract DINO features
python extract_feature.py --output_dir ../datasets/STEPFiles

# Encode SLat latents
python encode_latent.py --output_dir ../datasets/STEPFiles

# Update metadata after each step
python build_metadata.py STEPFiles --output_dir ../datasets/STEPFiles --source_dir ../step_files
```

### Step 4: Train UDF Decoder

```bash
cd /path/to/TRELLIS
python -u train.py \
    --config configs/vae/slat_vae_dec_udf_swin8_B_64l8_fp16.json \
    --output output/udf_decoder \
    --data_dir datasets/STEPFiles
```

## render_step.py Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Directory with processed STEP data |
| `--num_views` | int | 150 | Number of views to render per instance |
| `--renderer` | str | auto-detect | Path to `step_render` executable |
| `--instances` | str | None | Specific instances to process (comma-separated or file) |
| `--source_dir` | str | None | Source STEP file directory (required by STEPFiles dataset module) |
| `--rank` | int | 0 | Shard index for distributed processing |
| `--world_size` | int | 1 | Total number of shards |
| `--max_workers` | int | 4 | Parallel rendering workers |

### Per-Instance Processing

1. **Camera generation**: Generates `num_views` camera positions on a sphere (radius=2.0, FOV=40°) using Hammersley sequence with random offset for uniform coverage.

2. **Rendering**: Calls `step_render` binary with STEP file path and views JSON. Produces 512×512 RGBA PNGs.

3. **transforms.json**: Writes NeRF-compatible camera-to-world matrices. Feature extraction reads this to project DINOv2 features into 3D.

4. **Mesh conversion**: Converts `meshes/{name}.obj` → `renders/{name}/mesh.ply` (required by `voxelize.py`).

The script auto-skips instances that already have `transforms.json` with enough frames.

## prep_step_dataset.py Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Directory to save processed data |
| `--source_dir` | str | — | Directory containing source STEP files (initial setup only) |
| `--num_samples` | int | 100000 | Number of surface points to sample per instance |
| `--density` | float | 100.0 | Mesh density — target edge length = bbox_diagonal / density |
| `--resolution` | int | 256 | Voxel grid resolution for UDF normalization |
| `--edge_bias` | float | 0.5 | Fraction of samples biased toward near-edge triangles (0 = uniform, 1 = all near-edge) |
| `--edge_threshold` | float | 2.0 | UDF threshold in voxel widths defining "near-edge" faces |
| `--instances` | str | None | Specific instances to process (comma-separated sha256 values, or path to a file with one per line) |
| `--rank` | int | 0 | Shard index for distributed processing |
| `--world_size` | int | 1 | Total number of shards |
| `--max_workers` | int | 1 | Parallel workers (default 1; gmsh is not thread-safe) |

### Processing Steps (per file)

1. **Mesh generation**: Opens STEP file with gmsh, sets target edge length from `--density`, generates 2D surface mesh with fallback algorithms (Frontal-Delaunay → MeshAdapt → Delaunay).

2. **B-Rep edge extraction**: Reads gmsh line elements (1D edges on B-Rep boundaries) and maps them to vertex index pairs.

3. **Mesh cleanup**: Removes unreferenced vertices, remaps edge indices.

4. **Normalization**: Centers mesh at origin, scales to fit [-0.5, 0.5]³ with 5% margin (TRELLIS convention).

5. **Geodesic UDF**: Computes multi-source geodesic distance from all B-Rep edge vertices using the heat method (`potpourri3d`). Clamps to ≥ 0 (heat method can produce small negative artifacts at source vertices). Divides by `voxel_size = 1/resolution` so UDF=1.0 = one voxel width.

6. **Surface point sampling**: Samples `--num_samples` points on the mesh surface using area-weighted barycentric sampling with edge-biased distribution. Interpolates per-vertex UDF to sample points via barycentric coordinates.

### Edge-Biased Sampling

By default, 50% of sampled points are drawn preferentially from triangles near B-Rep edges (mean vertex UDF < `--edge_threshold`), and 50% are drawn uniformly by area. This gives much denser supervision in the near-edge region where UDF values change rapidly and precision matters most.

Setting `--edge_bias 0` disables biasing and reverts to pure area-weighted uniform sampling.

### Output Format

Each instance produces `udf/{sha256}.npz` containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `points` | [P, 3] | float32 | Surface point positions in [-0.5, 0.5]³ |
| `udf` | [P] | float32 | UDF values (voxel-normalized, ≥ 0) |
| `scale` | [1] | float32 | Scale factor applied during normalization |
| `offset` | [3] | float32 | Centroid offset applied during normalization |
| `num_edges` | [1] | int32 | Number of B-Rep edges found |

### UDF Value Interpretation

- **UDF = 0**: Point lies exactly on a B-Rep edge
- **UDF = 1**: Point is one voxel width (1/256 at default resolution) from the nearest edge
- **UDF > 1**: Point is farther from edges (interior of faces)

Typical ranges: min is always 0 (at edge vertices), max varies by geometry (10–70+ voxel widths for large flat faces).

## Dataset Directory Structure

After full processing, `datasets/STEPFiles/` contains:

```
datasets/STEPFiles/
├── metadata.csv                     # Master index
├── raw/                             # Symlinked source STEP files
├── meshes/                          # Normalized OBJ meshes
├── udf/                             # Surface points + UDF ground truth (.npz)
├── renders/                         # Multiview rendered images
├── voxels/                          # Voxelized occupancy grids
├── features/                        # DINOv2 sparse voxel features
└── latents/                         # Encoded SLat latents
    └── dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/
```

## Current Dataset

3 STEP file instances:

| Instance | Voxels | B-Rep Edges | Source |
|----------|--------|-------------|--------|
| aisin_part | 3,971 | ~200+ | Aisin automotive part |
| hook | 12,128 | ~100+ | Hook geometry |
| part | 7,151 | ~150+ | Generic mechanical part |
