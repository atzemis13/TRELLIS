---

## Parasolid Integration — COMPLETE

All items completed as of 2026-04-06. The pipeline has been fully migrated from gmsh/OCCT to Parasolid/NMR.

### What was done

1. **`prep_parasolid_dataset.py`** (renamed from `prep_step_dataset.py`) consumes Parasolid NPZ files. gmsh is no longer used.

2. **`DATASET_PARASOLID.md`** updated to reflect the Parasolid pipeline. All gmsh references removed.

3. **Verified** the Parasolid pipeline against gmsh on 41 ABC parts. Results: 80% agreement (EMD ≤ 5), median EMD = 0.68. The remaining 20% divergence is explained by OCCT's phantom seam edges on periodic surfaces (cylinders, cones) — the Parasolid topology is correct.

### Files renamed

- `prep_step_dataset.py` → `prep_parasolid_dataset.py`
- `render_step.py` → `render_parasolid.py`
- `datasets/STEPFiles.py` → `datasets/ParasolidFiles.py`
