# step_renderer: B-Rep Edge Curve Issues on ABC Dataset Parts

## Problem

Some ABC dataset STEP files have B-Rep edge curves whose underlying parametric definition extends far beyond the actual trimmed face boundaries. This causes:
1. **Stray lines** when OCCT draws face boundary edges
2. **Inflated bounding box** from `BRepBndLib::Add()` and `FitAll()`

The produced `.obj` meshes (via gmsh) are always correct. FreeCAD also displays these STEP files correctly.

## Root Cause

### 1. Stray lines from `SetFaceBoundaryDraw(True)`

OCCT's `AIS_Shape` draws face boundary edges by evaluating the full 3D edge curve. For some STEP files, these curves extend well past the face trim bounds. The rendered triangles (shaded faces) are fine — tessellation vertices are always within bounds.

**Why gmsh doesn't have this problem:** `mesh.generate(2)` tessellates face surfaces into triangles. Vertices are placed within the trimmed face. Edge curves are never drawn.

**Why FreeCAD doesn't have this problem:** FreeCAD uses Coin3D/pivy to render edges. Coin3D discretizes edge curves by evaluating between the edge's `First` and `Last` parameter values, which respect trim bounds. OCCT's `SetFaceBoundaryDraw` appears to not clip properly.

### 2. Inflated bounding box from `BRepBndLib::Add`

`BRepBndLib::Add()` computes the bbox from the B-Rep's underlying curve/surface parametric definitions, not from trimmed boundaries. This also affects `FitAll()` internally.

### ABC dataset specifics

Many ABC parts have edge curves (B-splines, lines) whose parametric domain extends beyond the trimmed portion used by faces. Some also have multi-body STEPs (part + stock blank).

## Current Solution

### External normalization (coordinate alignment with mesh/UDF)

Instead of computing our own bounding box from the STEP geometry (which is unreliable due to the edge curve issue), we pass the **mesher's exact normalization parameters** to the renderer:

- `prep_step_dataset.py` (gmsh mesher) computes `scale` and `offset` (centroid) from the tessellated mesh vertices and stores them in the UDF `.npz` file
- `render_step.py` reads `scale`/`offset` from the `.npz` and passes them to the renderer via `--scale`/`--offset` CLI args
- `render.cpp` applies the same transform: `(vertex - offset) * scale`

This guarantees the STEP render, mesh, and UDF all share **identical normalized coordinates** in [-0.5, 0.5]³.

### Fixed camera (no FitAll)

We use a fixed camera setup instead of `FitAll()`:
- Camera at radius 2.0, FOV 40°, looking at origin
- `SetZRange(0.1, 100.0)` for clipping
- `SetAutoZFitMode(False)` to prevent OCCT from overriding

This avoids the inflated B-Rep bbox affecting camera framing. Objects normalized to [-0.5, 0.5]³ fill ~30-50% of the 1024×1024 frame, which is consistent across all files.

### Edge rendering: stray lines accepted

`SetFaceBoundaryDraw(True)` is enabled because the TRELLIS encoder benefits from seeing B-Rep edges. Some files will have stray lines extending beyond the object, but:
- The stray lines are thin (1px) and few per file
- The object geometry and shading are always correct
- The fixed camera prevents the stray lines from shrinking the object (unlike `FitAll()`)

## What Was Attempted (and didn't work)

| Attempt | Result |
|---------|--------|
| `SetFaceBoundaryDraw(False)` | Removes stray lines but loses edge information |
| `FitAll(tightBox)` with explicit Bnd_Box | Blank output |
| Remove `FitAll()`, set `SetZRange()` manually | Initially blank — needed `SetAutoZFitMode(False)` |
| `ExtractLargestSolid()` | Not general — bad curves can be in the main solid |
| Extract faces only (discard wire bodies) | Doesn't help — stray lines come from face boundary edges |
| `BRepBndLib::AddOptimal()` | Still inflated |
| Tessellation-based bbox for `FitAll()` override | `FitAll()` still queries B-Rep bbox internally |

## Key Insight

The only reliable source of tight bounding box for these STEP files is **tessellation** (gmsh or `BRepMesh_IncrementalMesh` + `Poly_Triangulation` node iteration). The B-Rep curve/surface parametric definitions cannot be trusted for spatial bounds on ABC dataset files.

By using the mesher's normalization as the single source of truth, we sidestep the entire B-Rep bbox issue and guarantee coordinate alignment between all pipeline stages (mesh, UDF, renders).
