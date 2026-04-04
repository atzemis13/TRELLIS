---

## Parasolid Integration — Remaining Work

### Context

The Parasolid faceter produces `{id}.npz` files containing triangle meshes with B-Rep edge vertices identified (arrays: `vertices`, `faces`, `edge_vertices`). These replace gmsh as the meshing stage.

### What needs to happen

1. **Rewrite prep_step_dataset.py** to consume Parasolid NPZ files instead of calling gmsh. Drop gmsh entirely — no fallback. The script should load the faceted NPZ, normalize the mesh, compute geodesic UDF via the heat method, sample surface points, and save the same output format (meshes/*.obj + udf/*.npz). The multiprocessing can be simplified since there's no more gmsh instability to work around.

2. **Update DATASET_STEP.md** to reflect the new two-stage pipeline: Parasolid faceting (external) → prep_step_dataset.py (consumes NPZs). Remove gmsh references.

3. **Verify** the downstream pipeline is unaffected — the output format (OBJ meshes, UDF NPZs with points/udf/scale/offset/num_edges) stays the same, so render_step, voxelize, extract_feature, encode_latent, and training should all work unchanged. You can do this verification with just a few items from ABC, don't go crazy. 