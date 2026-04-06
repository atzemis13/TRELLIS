# Parasolid Dataset Preparation

This document covers data preparation for **Parasolid (`.x_t`) CAD files** used to train the UDF decoder. It is an addendum to [DATASET.md](DATASET.md), which covers the standard TRELLIS pipeline for mesh/image datasets.

## Overview

The UDF decoder predicts geodesic distance from each grid vertex to the nearest B-Rep (CAD) edge. Training data is generated from Parasolid (`.x_t`) files through a two-stage pipeline:

**Stage 1 — Parasolid faceting (external, via NMR):**
1. Tessellates the B-Rep geometry into a deduplicated triangle mesh
2. Identifies which vertices lie on B-Rep edges (excluding seam edges)
3. Outputs `{sha256}.npz` files with `vertices`, `faces`, and `edge_vertices` arrays

**Stage 2 — UDF computation (`prep_parasolid_dataset.py`):**
1. Loads Stage-1 NPZs
2. Normalizes the mesh to [-0.5, 0.5]³
3. Computes geodesic distance fields (heat method) from B-Rep edge vertices
4. Samples surface points with edge-biased distribution
5. Produces `.npz` files compatible with the `SLat2UDF` dataset class

After Parasolid-specific processing, the standard TRELLIS pipeline (render, voxelize, extract features, encode latents) is run as described in [DATASET.md](DATASET.md).

## Prerequisites

**Stage 1 (NMR):** Build `nmr-server` from `/Users/charlie/atomic/nmr` — see `nmr/local-build.md`.

**Stage 2 (Python):** Standard TRELLIS environment plus:

```bash
pip install potpourri3d trimesh
```

## Pipeline

### Step 0: Prepare Parasolid Files

Place Parasolid (`.x_t`) files in a source directory (e.g., `parasolid_files/`).

### Step 1a: Facet with Parasolid (NMR)

For each `.x_t` file, run `nmr-server --convert` to produce a mesh NPZ. The output filename should be `{sha256}.npz` where `sha256` is the SHA-256 of the source file.

```bash
NMR=/path/to/nmr/build/nmr-server
DYLD_LIBRARY_PATH=/path/to/nmr/parasolid/shared_object  # macOS only
NPZ_DIR=/path/to/parasolid_npz

for f in parasolid_files/*.x_t; do
    sha=$(shasum -a 256 "$f" | awk '{print $1}')
    $NMR --convert "$f" --tess-max-width 0.02 -o "$NPZ_DIR/${sha}.npz"
done
```

This produces for each file:
- `{sha256}.npz` — Deduplicated triangle mesh with `vertices` [V,3], `faces` [F,3], `edge_vertices` [E]

### Step 1b: Compute UDF (Python)

```bash
cd dataset_toolkits
python prep_parasolid_dataset.py ParasolidFiles \
    --source_dir /path/to/parasolid_files \
    --npz_dir /path/to/parasolid_npz \
    --output_dir datasets/ParasolidFiles
```

This generates for each instance:
- `meshes/{sha256}.obj` — Normalized triangulated mesh in [-0.5, 0.5]³
- `udf/{sha256}.npz` — Surface points with ground truth UDF values
- `metadata.csv` — Dataset metadata index

### Step 2: Render Multiview Images

Parasolid files are rendered using `render_parasolid.py` with the NMR-based `parasolid_renderer.py` instead of the standard Blender-based `render.py`.

```bash
cd dataset_toolkits
python render_parasolid.py ParasolidFiles \
    --output_dir ../datasets/ParasolidFiles \
    --source_dir ../parasolid_files \
    --num_views 150 \
    --max_workers 4
```

This generates for each instance:
- `renders/{name}/000.png` … `149.png` — RGBA multiview images (1024×1024)
- `renders/{name}/transforms.json` — Camera matrices (NeRF-style, needed by feature extraction)
- `renders/{name}/views.json` — Camera parameters (yaw, pitch, radius, fov)
- `renders/{name}/mesh.ply` — Mesh converted from OBJ (needed by voxelization)

### Step 3: Standard TRELLIS Pipeline (Voxelize → Features → Latents)

After rendering, run the remaining standard pipeline steps from [DATASET.md](DATASET.md):

```bash
cd dataset_toolkits

# Voxelize
python voxelize.py ParasolidFiles --output_dir ../datasets/ParasolidFiles --source_dir ../parasolid_files

# Extract DINO features
python extract_feature.py --output_dir ../datasets/ParasolidFiles

# Encode SLat latents
python encode_latent.py --output_dir ../datasets/ParasolidFiles

# Update metadata after each step
python build_metadata.py ParasolidFiles --output_dir ../datasets/ParasolidFiles --source_dir ../parasolid_files
```

### Step 4: Train UDF Decoder

```bash
cd /path/to/TRELLIS
python -u train.py \
    --config configs/vae/slat_vae_dec_udf_swin8_B_64l8_fp16.json \
    --output output/udf_decoder \
    --data_dir datasets/ParasolidFiles
```

## render_parasolid.py Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Directory with processed CAD data |
| `--num_views` | int | 150 | Number of views to render per instance |
| `--renderer` | str | auto-detect | Path to parasolid_renderer.py |
| `--npz_dir` | str | None | Directory of pre-built NMR NPZ files (for parasolid_renderer.py) |
| `--instances` | str | None | Specific instances to process (comma-separated or file) |
| `--source_dir` | str | None | Source Parasolid file directory (required by ParasolidFiles dataset module) |
| `--rank` | int | 0 | Shard index for distributed processing |
| `--world_size` | int | 1 | Total number of shards |
| `--max_workers` | int | 4 | Parallel rendering workers |

## prep_parasolid_dataset.py Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Directory to save processed data |
| `--npz_dir` | str | None | Directory containing Parasolid NPZ files (Stage-1 output from NMR) |
| `--num_samples` | int | 100000 | Number of surface points to sample per instance |
| `--resolution` | int | 256 | Voxel grid resolution for UDF normalization |
| `--edge_bias` | float | 0.5 | Fraction of samples biased toward near-edge triangles (0 = uniform, 1 = all near-edge) |
| `--edge_threshold` | float | 2.0 | UDF threshold in voxel widths defining "near-edge" faces |
| `--instances` | str | None | Specific instances to process (comma-separated sha256 values, or path to a file with one per line) |
| `--rank` | int | 0 | Shard index for distributed processing |
| `--world_size` | int | 1 | Total number of shards |
| `--max_workers` | int | 4 | Parallel workers |

### Output Format

Each instance produces `udf/{sha256}.npz` containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `points` | [P, 3] | float32 | Surface point positions in [-0.5, 0.5]³ |
| `udf` | [P] | float32 | UDF values (voxel-normalized, ≥ 0) |
| `scale` | [1] | float32 | Scale factor applied during normalization |
| `offset` | [3] | float32 | Centroid offset applied during normalization |
| `num_edges` | [1] | int32 | Number of B-Rep edges found |

## Dataset Directory Structure

After full processing, `datasets/ParasolidFiles/` contains:

```
datasets/ParasolidFiles/
├── metadata.csv                     # Master index
├── raw/                             # Symlinked source Parasolid files
├── meshes/                          # Normalized OBJ meshes
├── udf/                             # Surface points + UDF ground truth (.npz)
├── renders/                         # Multiview rendered images
├── voxels/                          # Voxelized occupancy grids
├── features/                        # DINOv2 sparse voxel features
└── latents/                         # Encoded SLat latents
    └── dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/
```
