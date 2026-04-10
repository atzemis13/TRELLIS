# Scaling the UDF Training Pipeline to 100k Parts

Notes from the 5-part overfit experiment (April 2026).

## Pipeline Overview

The full pipeline for each part:

1. **NMR convert** — `nmr --convert input.x_t -o {sha256}.npz` → vertices, faces, edge_vertices
2. **UDF prep** — `prep_parasolid_dataset.py` → geodesic UDF from heat method, surface point sampling
3. **Render** — `render_parasolid.py` → 150 multi-view images via NMR Qt `--snapshot`
4. **Voxelize** — `voxelize.py` → 64³ voxel grid
5. **Extract features** — `extract_feature.py` → DINOv2 ViT-L/14 patch tokens projected to voxels
6. **Encode latents** — `encode_latent.py` → SLAT latents via pretrained encoder
7. **Build metadata** — `build_metadata.py --from_file` → update metadata.csv with all flags
8. **Train** — `train.py` with `inflat_all` FP16 mode

Steps 1-4 are CPU-only (need Parasolid/NMR). Steps 5-6 need GPU. Step 8 needs GPU.

## Time Estimates per Part (single-threaded)

| Step | Time/part | 100k total | Parallelizable |
|------|-----------|------------|----------------|
| NMR convert | ~0.5s | ~14h | Yes (CPU) |
| UDF prep (geodesic) | ~0.2s | ~6h | Yes (CPU, `--max_workers`) |
| Render (150 views) | ~3s | ~80h | Yes (CPU, `--max_workers`, `--rank/--world_size`) |
| Voxelize | ~0.1s | ~3h | Yes (CPU) |
| DINOv2 features | ~7s | ~20h | Shard across GPUs (`--rank/--world_size`) |
| Encode latents | ~1s | ~3h | Shard across GPUs |
| Training (10k steps) | ~2h total | ~2h | Multi-GPU (`--num_gpus`) |

**Rendering is the main bottleneck.** 80 hours single-threaded. Options:
- `--max_workers 8` on a beefy CPU machine → ~10h
- Shard across N machines with `--rank 0..N-1 --world_size N`
- Reduce `--num_views` (150 is the default; fewer views = less feature quality)

## Known Issues & Fixes

### render_parasolid.py offset format bug
Scientific notation in offset values (e.g., `-3.6e-05`) was parsed by argparse as a flag.
**Fixed:** format offsets as `f'{value:.10f}'` instead of `str(value)`.
This fix is committed — make sure it's deployed before large-scale rendering.

### File descriptor limit on high-core machines
DataLoader spawns one worker per CPU core. On 128-core machines, this exceeds the default `ulimit -n 1024`.
**Fix:** `ulimit -n 65536` before training.

### GPU environment must use cu118 stack
The `inflat_all` FP16 training mode produces gradient collapse (gradients underflow to zero) when using PyTorch cu124 + spconv-cu124. 
**Fix:** Always use `setup_uv.sh` which installs PyTorch 2.4.0+cu118 + spconv-cu118. This is the tested combo. See `setup_uv.sh` for the full install procedure.

### flash-attn prebuilt wheel
`setup_uv.sh` tries to compile flash-attn, which fails if the host CUDA toolkit doesn't match cu118. Download the prebuilt wheel instead:
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install <wheel> --no-deps
```

### FlexiCubes submodule
Not auto-cloned by rsync. Must be cloned manually on the target:
```bash
git clone https://github.com/MaxtirError/FlexiCubes.git trellis/representations/mesh/flexicubes
```

## Data Format Reference

**x_t files → SHA256 keys:** The `ParasolidFiles` dataset helper hashes x_t file content to produce SHA256 identifiers. These match the NPZ filenames in `npz_files/`.

**UDF normalization:** Geodesic distance is divided by `voxel_size = 1/256`. So UDF=1.0 means one voxel width from the nearest B-Rep edge. Values can reach 300+ for large flat faces far from edges. This is the same normalization as the old gmsh pipeline.

**Dataset directory layout:**
```
datasets/{name}/
├── metadata.csv          # Master index with boolean flags per instance
├── raw/{sha256}.x_t      # Symlinks to source CAD files
├── meshes/{sha256}.obj   # Normalized mesh in [-0.5, 0.5]³
├── udf/{sha256}.npz      # points [P,3] + udf [P] + scale/offset
├── renders/{sha256}/     # 150 PNGs + transforms.json + mesh.ply
├── voxels/{sha256}.ply   # 64³ voxel center positions
├── features/{model}/     # DINOv2 patch tokens per voxel
└── latents/{model}/      # SLAT latents (coords + feats)
```

## Training Config for Overfit

See `configs/vae/overfit5_udf.json`. Key fields to change for full training:
- `dataset.args.include_instances` — remove or set to all instances
- `trainer.args.max_steps` — increase from 10k
- `trainer.args.batch_size_per_gpu` / `batch_split` — tune for GPU memory
- `trainer.args.i_save` — checkpoint frequency

## Cloud Deploy Checklist

1. Provision GPU instance (any image works)
2. `rsync` repo (exclude `.venv`, `.git`, `__pycache__`)
3. `apt-get install -y rsync` (if needed for bidirectional sync)
4. `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. Clone FlexiCubes submodule
6. `source setup_uv.sh --new-env --basic --train --flash-attn --spconv --kaolin`
7. Install flash-attn prebuilt wheel if compile failed
8. `ulimit -n 65536`
9. `source .venv/bin/activate`
10. Run pipeline scripts
