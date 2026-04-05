#!/usr/bin/env python3
"""
Prepare Parasolid (STEP) files for TRELLIS UDF decoder training.

This script is Stage 2 of a two-stage pipeline:
  Stage 1 (external): nmr --convert input.x_t -o {sha256}.npz
                       Produces a deduplicated triangle mesh with B-Rep edge
                       vertices identified (arrays: vertices, faces, edge_vertices).
  Stage 2 (this script): Loads Stage-1 NPZs, normalizes the mesh, computes
                          geodesic UDF via the heat method, samples surface
                          points, and saves the output format expected by
                          the SLat2UDF dataset class.

Output per instance:
  meshes/{sha256}.obj   — Normalized triangulated mesh in [-0.5, 0.5]^3
  udf/{sha256}.npz      — Surface points with ground-truth UDF values

Usage:
    python prep_step_dataset.py STEPFiles \
        --npz_dir /path/to/parasolid_npz \
        --output_dir datasets/STEPFiles
"""

import os
import sys
import importlib
import argparse
import copy
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from multiprocessing import Pool
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def load_parasolid_npz(npz_path):
    """
    Load a Parasolid-generated NPZ file.

    Returns:
        vertices:      np.ndarray [V, 3] float64
        faces:         np.ndarray [F, 3] int32
        edge_vertices: np.ndarray [E]    int32  (indices of B-Rep edge verts)
    """
    data = np.load(npz_path)
    vertices = data['vertices']          # [V, 3] float64
    faces = data['faces']                # [F, 3] int32
    edge_vertices = data['edge_vertices']  # [E]  int32
    return vertices, faces, edge_vertices


def normalize_mesh(vertices, faces):
    """
    Normalize mesh to [-0.5, 0.5]^3 (TRELLIS convention).

    Returns:
        norm_vertices: np.ndarray [V, 3] float32
        scale:         float
        offset:        np.ndarray [3] float64  (original centroid)
    """
    verts = vertices.copy()
    centroid = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts -= centroid

    max_extent = np.abs(verts).max()
    scale = (0.5 / max_extent * 0.95) if max_extent > 0 else 1.0
    verts *= scale

    return verts.astype(np.float32), float(scale), centroid


def compute_geodesic_udf(vertices, faces, edge_vertex_indices):
    """
    Compute geodesic distance from each mesh vertex to the nearest B-Rep
    edge vertex using the heat method (potpourri3d).

    Returns per-vertex UDF values (float32, ≥ 0).
    """
    import potpourri3d as pp3d

    if len(edge_vertex_indices) == 0:
        print("  Warning: no B-Rep edge vertices found, returning zero UDF")
        return np.zeros(len(vertices), dtype=np.float32)

    source_verts = np.array(sorted(set(edge_vertex_indices.tolist())), dtype=np.int32)
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    dist = solver.compute_distance_multisource(source_verts)
    # Clamp: heat method can produce tiny negatives at source vertices
    return np.maximum(dist, 0.0).astype(np.float32)


def sample_surface_with_udf(vertices, faces, vertex_udf,
                             n_samples, edge_bias=0.5, edge_threshold=2.0):
    """
    Sample points on the mesh surface with interpolated UDF values.
    Uses edge-biased sampling so that near-edge regions (low UDF) get
    denser supervision.

    Returns:
        points: [n_samples, 3] float32
        udf:    [n_samples]    float32
    """
    # Per-face area (cross-product method)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area <= 0:
        total_area = 1.0

    # Per-face mean UDF
    face_mean_udf = vertex_udf[faces].mean(axis=1)
    near_edge_mask = face_mean_udf < edge_threshold
    n_near_edge = near_edge_mask.sum()

    if n_near_edge > 0 and edge_bias > 0:
        n_edge_samples    = int(n_samples * edge_bias)
        n_uniform_samples = n_samples - n_edge_samples

        # Near-edge: area-weighted over near-edge faces only
        edge_areas = areas.copy()
        edge_areas[~near_edge_mask] = 0.0
        edge_probs = edge_areas / edge_areas.sum()
        edge_tri_indices = np.random.choice(len(faces), size=n_edge_samples, p=edge_probs)

        # Uniform: area-weighted over all faces
        uniform_probs = areas / total_area
        uniform_tri_indices = np.random.choice(len(faces), size=n_uniform_samples,
                                               p=uniform_probs)

        tri_indices = np.concatenate([edge_tri_indices, uniform_tri_indices])
        print(f"  Edge-biased sampling: {n_near_edge}/{len(faces)} near-edge faces "
              f"({n_edge_samples} edge + {n_uniform_samples} uniform = {n_samples} total)")
    else:
        uniform_probs = areas / total_area
        tri_indices = np.random.choice(len(faces), size=n_samples, p=uniform_probs)
        if n_near_edge == 0:
            print(f"  Warning: no near-edge faces (threshold={edge_threshold}), "
                  f"using uniform sampling")

    # Barycentric coordinates
    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    bary = np.stack([u, v, w], axis=1)  # [N, 3]

    tri_verts = vertices[faces[tri_indices]]              # [N, 3, 3]
    points = np.einsum('ijk,ij->ik', tri_verts, bary)    # [N, 3]

    corner_udf = vertex_udf[faces[tri_indices]]           # [N, 3]
    udf = np.einsum('ij,ij->i', corner_udf, bary)        # [N]

    perm = np.random.permutation(n_samples)
    return points[perm].astype(np.float32), udf[perm].astype(np.float32)


def _process_instance(npz_path, sha256, output_dir,
                      num_samples, resolution=256,
                      edge_bias=0.5, edge_threshold=2.0):
    """Process a single Parasolid NPZ file."""
    import trimesh

    mesh_dir = os.path.join(output_dir, 'meshes')
    udf_dir  = os.path.join(output_dir, 'udf')
    mesh_path = os.path.join(mesh_dir, f'{sha256}.obj')
    udf_path  = os.path.join(udf_dir,  f'{sha256}.npz')

    if os.path.exists(mesh_path) and os.path.exists(udf_path):
        return {'sha256': sha256, 'has_mesh': True, 'has_udf': True}

    try:
        vertices, faces, edge_vertices = load_parasolid_npz(npz_path)

        if len(faces) == 0:
            print(f"  Warning: no faces in {sha256}")
            return None

        # Normalize to [-0.5, 0.5]^3
        norm_verts, scale, offset = normalize_mesh(vertices, faces)

        # Save OBJ mesh
        mesh_obj = trimesh.Trimesh(vertices=norm_verts, faces=faces, process=False)
        mesh_obj.export(mesh_path)

        # Compute geodesic UDF on the normalized mesh
        vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)

        # Normalize UDF by voxel size: UDF=1.0 means one voxel width from nearest edge
        voxel_size = 1.0 / resolution
        norm_udf = vertex_udf / voxel_size

        # Adaptive sample count: scale with mesh complexity
        n_faces = len(faces)
        if num_samples == 100000:
            adaptive_samples = max(50000, min(1500000, n_faces * 5))
        else:
            adaptive_samples = num_samples

        points, udf = sample_surface_with_udf(
            norm_verts, faces, norm_udf, adaptive_samples,
            edge_bias=edge_bias, edge_threshold=edge_threshold)

        np.savez(
            udf_path,
            points=points,
            udf=udf,
            scale=np.array([scale], dtype=np.float32),
            offset=offset.astype(np.float32),
            num_edges=np.array([len(edge_vertices)], dtype=np.int32),
        )

        return {
            'sha256':       sha256,
            'has_mesh':     True,
            'has_udf':      True,
            'num_faces':    n_faces,
            'num_vertices': len(vertices),
            'num_edge_verts': int(len(edge_vertices)),
            'num_samples':  adaptive_samples,
        }

    except Exception as e:
        print(f"  Error processing {sha256}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _worker_fn(args):
    return _process_instance(
        npz_path=args['npz_path'],
        sha256=args['sha256'],
        output_dir=args['output_dir'],
        num_samples=args['num_samples'],
        resolution=args['resolution'],
        edge_bias=args['edge_bias'],
        edge_threshold=args['edge_threshold'],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--npz_dir', type=str, default=None,
                        help='Directory containing Parasolid NPZ files '
                             '(output of "nmr --convert input.x_t -o {sha256}.npz")')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of surface points to sample for UDF')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Voxel grid resolution for UDF normalization')
    parser.add_argument('--edge_bias', type=float, default=0.5,
                        help='Fraction of samples biased toward near-edge triangles')
    parser.add_argument('--edge_threshold', type=float, default=2.0,
                        help='UDF threshold (voxel widths) for near-edge triangles')
    parser.add_argument('--instances', type=str, default=None,
                        help='Specific instances to process (comma-separated sha256 or file)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel worker processes')
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'udf'),    exist_ok=True)

    metadata_path = os.path.join(opt.output_dir, 'metadata.csv')

    # Build or load metadata
    if not os.path.exists(metadata_path):
        print('Building metadata from source directory...')
        metadata = dataset_utils.get_metadata(**opt)
        metadata.to_csv(metadata_path, index=False)
        print(f'Created metadata.csv with {len(metadata)} instances')
        print('Linking source files...')
        downloaded = dataset_utils.download(metadata, opt.output_dir)
        print(f'Linked {len(downloaded)} files')
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.read_csv(metadata_path)
        if getattr(opt, 'source_dir', None) is not None:
            try:
                new_metadata = dataset_utils.get_metadata(**opt)
                existing_ids = set(metadata['sha256'].values)
                new_rows = new_metadata[~new_metadata['sha256'].isin(existing_ids)]
                if len(new_rows) > 0:
                    print(f'Found {len(new_rows)} new files in {opt.source_dir}')
                    downloaded = dataset_utils.download(new_rows, opt.output_dir)
                    print(f'Linked {len(downloaded)} new files')
                    metadata = pd.concat([metadata, new_rows], ignore_index=True)
                    metadata.to_csv(metadata_path, index=False)
                    print(f'Updated metadata.csv: {len(metadata)} total instances')
            except Exception as e:
                print(f'Warning: could not scan source_dir for updates: {e}')

    # Filter instances
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances) as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Shard
    start = len(metadata) * opt.rank // opt.world_size
    end   = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # Skip already-processed
    records = []
    for sha256 in copy.copy(metadata['sha256'].values):
        mesh_path = os.path.join(opt.output_dir, 'meshes', f'{sha256}.obj')
        udf_path  = os.path.join(opt.output_dir, 'udf',    f'{sha256}.npz')
        if os.path.exists(mesh_path) and os.path.exists(udf_path):
            records.append({'sha256': sha256, 'has_mesh': True, 'has_udf': True})
            metadata = metadata[metadata['sha256'] != sha256]

    # Build work items — locate Parasolid NPZ for each instance
    npz_dir = opt.npz_dir
    work_items = []
    skipped_missing = 0
    for _, row in metadata.iterrows():
        sha256 = row['sha256']

        # Look for {sha256}.npz in the npz_dir
        if npz_dir:
            npz_path = os.path.join(npz_dir, f'{sha256}.npz')
        else:
            npz_path = None

        if npz_path and os.path.exists(npz_path):
            work_items.append({
                'npz_path':     npz_path,
                'sha256':       sha256,
                'output_dir':   opt.output_dir,
                'num_samples':  opt.num_samples,
                'resolution':   opt.resolution,
                'edge_bias':    opt.edge_bias,
                'edge_threshold': opt.edge_threshold,
            })
        else:
            print(f"Parasolid NPZ not found for {sha256}: {npz_path}")
            skipped_missing += 1

    n_workers = min(opt.max_workers, len(work_items)) if work_items else 1
    print(f'Processing {len(work_items)} instances with {n_workers} workers...')
    if skipped_missing:
        print(f'  Skipped {skipped_missing} instances with missing NPZ files')

    processed_records = []
    if n_workers <= 1:
        for item in tqdm(work_items, desc='Processing'):
            rec = _worker_fn(item)
            if rec is not None:
                processed_records.append(rec)
    else:
        with Pool(processes=n_workers) as pool:
            for rec in tqdm(pool.imap_unordered(_worker_fn, work_items),
                            total=len(work_items), desc='Processing'):
                if rec is not None:
                    processed_records.append(rec)

    # Save progress CSV
    processed = pd.DataFrame.from_records(processed_records + records)
    processed.to_csv(
        os.path.join(opt.output_dir, f'step_processed_{opt.rank}.csv'),
        index=False)

    n_success = len(processed_records)
    n_skipped = len(records)
    print(f'\nDone: {n_success} processed, {n_skipped} already existed')
    print(f'Total in output: {len(processed)}')
    print('Run build_metadata.py to update metadata.csv')
