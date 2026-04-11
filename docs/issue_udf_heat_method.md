# Issue: Heat Method Produces Inaccurate UDF at B-Rep Edges

## Summary

The geodesic distance computation in `prep_parasolid_dataset.py` uses `potpourri3d.MeshHeatMethodDistanceSolver` to compute distance from B-Rep edge vertices to all mesh vertices. **The heat method produces significant errors at and near the source (edge) vertices**, with ~50% of edge vertices having UDF > 0.1 voxel widths when they should be exactly zero. This corrupts the ground truth training data and limits the model's ability to learn sharp edge detection.

## Evidence

Tested on 5 parts (Block, Mug, Door Stop, Bike Clamp, Flange) with fine tessellation (`--tess-max-width 0.02`):

| Part | Edge Verts | UDF > 0.1 | Max UDF at Edge |
|------|-----------|-----------|-----------------|
| Mug | 2,368 | 52% | 3.42 |
| Block | 3,606 | 47% | 4.95 |
| Door Stop | 2,388 | 38% | 2.70 |
| Bike Clamp | 4,976 | 42% | 3.62 |
| Flange | 3,962 | 55% | 5.32 |

UDF values are normalized by voxel size (1/256), so UDF=1.0 means one voxel width. Values up to 5.3 voxel widths at supposed zero-distance locations.

Increasing `t_coef` (heat diffusion time) makes the problem worse, not better.

## Impact

- The UDF decoder trains on noisy GT where edge locations have non-zero UDF
- Predicted UDF shows correct spatial gradients (scatter plot MAE ~0.2) but the near-edge GT itself is wrong
- Painted mesh visualizations show edges in roughly the right places but with gaps and missing segments
- The issue exists in both the NMR/Parasolid pipeline and likely the old gmsh pipeline

## Root Cause

The heat method solves a Poisson equation to estimate geodesic distance. With many source vertices densely packed along B-Rep edges:

1. The triangles along edges are long and thin (constrained by tessellation placing vertices on B-Rep curves)
2. The PDE discretization is inaccurate on poorly-shaped triangles
3. Multi-source computation (all edge vertices simultaneously) causes interference between nearby sources
4. Both positive and negative distance errors occur (we clamp negatives to zero but positive errors pass through)

## Potential Fixes

### 1. Hybrid Euclidean/Geodesic
Use exact Euclidean distance to nearest B-Rep edge *segment* for vertices within some radius of an edge. Fall back to geodesic (heat method) for far-away vertices where it's accurate. Requires computing point-to-line-segment distances for all mesh edges connecting edge vertex pairs.

**Pros:** Exact at edges where it matters, geodesic where shape matters  
**Cons:** Need to pick a transition radius; Euclidean misses thin-wall cases where a geodesic edge is on the other side

### 2. Different Geodesic Solver
`potpourri3d` has `MeshFastMarchingDistanceSolver` which takes curve inputs (not just point sources). Passing the actual B-Rep edge segments as curves would compute distance to edges, not vertices, and might be more accurate. However, the API is poorly documented and initial attempts didn't produce correct results.

### 3. Remesh Before Computing
Run the tessellated mesh through a quality remesher to improve triangle aspect ratios, especially near edges. Then compute heat method on the better mesh and map distances back.

### 4. Post-hoc Correction
After computing heat method distances, detect and correct anomalous values at edge vertices. Could use local smoothing, or replace edge vertex values with the minimum of their non-edge neighbors' values minus the edge length.

## Test

See `tests/test_udf_at_edges.py`. Three tests:

1. **test_edge_vertices_have_near_zero_udf** — All edge vertices must have UDF < 0.1
2. **test_near_edge_faces_have_low_udf** — Vertices adjacent to edges must have UDF < 2.0
3. **test_sampled_edge_points_have_near_zero_udf** — Midpoints of B-Rep edge segments must have UDF < 0.2

All three currently **FAIL** with the heat method.
