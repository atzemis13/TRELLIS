# UDF Decoder Architecture Notes

Working document for architectural decisions on the UDF decoder. The decoder was adapted from the mesh decoder (`slat_dec_mesh_swin8_B_64l8m256c_fp16`) and may carry design choices that aren't justified for UDF.

## Current Architecture

Config: `configs/vae/slat_vae_dec_udf_swin8_B_64l8_fp16.json`

```
Input:  SLAT latents — sparse 3D, 64³ lattice, 8 channels
  ↓
SparseTransformerBase (shared with mesh decoder)
  - 12 blocks, 12 heads, 768 channels
  - Swin attention, window_size=8, FP16
  ↓
SparseSubdivideBlock3d × 2 (upsample 64 → 128 → 256)
  - 768 → 192 → 96 channels
  ↓
SparseLinear → 8 channels (one UDF per voxel corner)
  ↓
SparseFeatures2UDF
  - softplus(·) to enforce UDF ≥ 0
  - sparse_cube2verts: average 8 corner predictions at shared vertices (con_loss)
  - get_dense_attrs: scatter to dense 257³ grid, sentinel -1.0 for unoccupied
  - smoothness reg: mean |∇UDF| across valid vertex pairs (weight 0.01)
  ↓
Output: UDFExtractResult with udf_grid [257, 257, 257] + reg_loss
  ↓
interpolate_udf_to_points: grid_sample trilinear at query points
```

Parameters: 90.9M

## Inherited from Mesh Decoder (Unquestioned)

| Design choice | Mesh decoder reason | UDF necessity |
|---------------|---------------------|---------------|
| 257³ vertex grid | FlexiCubes needs corner values for marching cubes | Unclear — UDF is just a scalar field |
| 8 outputs per voxel (corners) | FlexiCubes needs SDF at 8 cube corners | See debate below |
| 64 → 256 resolution (2 upsamples) | Surface detail fidelity | Unclear for UDF |
| FP16 backbone | Memory savings | OK, no strong case against |
| SparseSubdivideBlock3d duplicated in both files | Copy-paste | — |
| Zero-init of out_layer | Training stability for both | Keep |

## Debate: 8-Corner Output Representation

### Arguments for keeping 8 corners (evaluated as strong)

1. **Implicit ensemble / variance reduction**
   - Interior vertices receive up to 8 predictions averaged at `sparse_cube2verts`
   - Even at correlation ρ=0.9, averaging gives ~40% variance reduction
   - The `con_loss` that penalizes disagreement between voxels is a principled smoothness regularizer, not overhead
   - Analogous to established multi-head prediction / auxiliary supervision patterns

2. **Intra-voxel gradient representation (Q₁ finite elements)**
   - 8 corner values define a trilinear interpolant with piecewise-bilinear gradient field
   - Cell-centered alternative gives piecewise-constant gradients only
   - For Eikonal supervision |∇d|=1, can be enforced at 12 edge midpoints + 6 face centers per voxel, all locally within one voxel
   - Cell-centered gradients smear across voxel boundaries where UDF has kinks

3. **Finite element approximation quality**
   - For a planar edge through a voxel, trilinear represents the linear UDF ramp exactly
   - Cell-centered worst-case error: h√3/2
   - Lipschitz approximation error: superlinear for Q₁, linear for piecewise-constant

4. **Alignment with literature**
   - DMTet, FlexiCubes, ConvONet all use vertex/corner storage
   - Finite element tradition places DOF at nodes for C⁰ continuity
   - This is a convergent choice across the field, not FlexiCubes-specific

5. **Future-proofing for level set extraction**
   - If we ever want to extract edge curves as zero level-set, marching-cubes-style needs corner values
   - Free given we're keeping the representation

### Arguments against (evaluated as weaker)

- Output layer is 8× larger than strictly needed for the raw UDF values alone
- Aggregation step (`sparse_cube2verts`) is compute overhead
- These costs could be absorbed elsewhere if we needed to reduce memory

### Current assessment

The 8-corner design has real theoretical justification for UDF prediction independent of mesh extraction. The ensemble + gradient arguments are the strongest. **Leaning: keep.**

Open question: does an ablation show measurable quality difference on real data?

## Question: 257³ Grid Resolution

The grid materializes 17M vertices to supervise ~50k surface points per part. Most of the grid represents empty space.

### Arguments for keeping 257³
- If we add Eikonal regularization, we want the field smooth everywhere, not just at training samples
- If we ever extract edge curves, we want spatial resolution
- Matches the mesh decoder backbone architecture (no rework needed)

### Arguments against
- Most UDF literature (DeepSDF, ConvONet, NeuralUDF, CAP-UDF) uses continuous point queries, never materializes a dense grid
- The dense 257³ with the +1 vertex ghost layer is primarily a mesh-extraction convenience
- 64³ (the native SLAT resolution) or 128³ might suffice if combined with proper interpolation
- Memory/compute on large-scale datasets could become a bottleneck

### Current assessment

More uncertain than the 8-corner question. The grid size is coupled to the upsampling structure (two 2× upsample blocks), so changing it involves changing the whole decoder topology. **Needs more thought.**

## Other Architectural Ideas (Not Yet Evaluated)

From the literature survey:

- **Log-space distance output**: predict `log(d + ε)` instead of `d`. Better precision near edges where it matters; heavy-tail suppression for far points.
- **Eikonal regularization**: enforce `|∇UDF| = 1` on the surface. Standard in SDF/UDF; aligned with what 8-corner representation enables.
- **Gradient prediction head**: auxiliary 3-vector output for ∇UDF, supervised via finite differences on GT. Provides edge direction signal.
- **Edge classification head**: binary "is this near an edge" with focal loss, alongside regression.
- **Multi-scale auxiliary losses**: supervise UDF at intermediate transformer layers.
- **Cross-attention point queries**: 3DShape2VecSet-style. Query points cross-attend to sparse latents; small MLP decodes. Eliminates dense grid entirely.
- **Truncated UDF (TUDF)**: clamp distance at some threshold T. Focuses capacity on near-edge region.

## Relevant File References

- `trellis/models/structured_latent_vae/decoder_udf.py` — decoder
- `trellis/models/structured_latent_vae/decoder_mesh.py` — mesh decoder (for comparison)
- `trellis/models/structured_latent_vae/base.py` — SparseTransformerBase
- `trellis/representations/mesh/udf.py` — SparseFeatures2UDF, `interpolate_udf_to_points`
- `trellis/representations/mesh/cube2mesh.py` — mesh representation (for comparison)
- `trellis/representations/mesh/utils_cube.py` — `sparse_cube2verts`, `get_dense_attrs`, `con_loss` (line 44)
- `trellis/trainers/vae/structured_latent_vae_udf_dec.py` — training loss

## Open Decisions

| Question | Status |
|----------|--------|
| Keep 8-corner output? | Leaning yes (strong theoretical case) |
| Keep 257³ grid? | Open — depends on whether we add Eikonal or switch to point queries |
| Add log-space output? | Worth trying |
| Add Eikonal regularization? | Worth trying, especially with 8-corner representation |
| Add gradient head? | Worth trying |
| Switch to cross-attention point queries? | Major refactor — defer until we have more data |
| Dedupe `SparseSubdivideBlock3d`? | Low priority cleanup |

Nothing is final. This is a working doc — we'll make decisions together.
