#!/usr/bin/env python3
"""
Prepare Parasolid files for TRELLIS UDF decoder training.

Two pipeline steps, both tracked in metadata.csv:

  1. Convert: x_t → converted/{sha256}.npz  (has_converted=True)
     Runs NMR with --tess-max-width to produce a triangle mesh with
     B-Rep edge vertices identified.

  2. UDF: converted NPZ → meshes/{sha256}.obj + udf/{sha256}.npz  (has_mesh=True, has_udf=True)
     Normalizes mesh, computes geodesic UDF via heat method, samples
     surface points.

Usage:
    python prep_parasolid_dataset.py ParasolidFiles \
        --source_dir /path/to/x_t_files \
        --output_dir datasets/ParasolidFiles

Both steps run in one invocation. Already-completed instances are skipped
based on filesystem checks.
"""

import os
import sys
import importlib
import argparse
import copy
import subprocess
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from multiprocessing import Pool
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: NMR conversion (x_t → NPZ)
# ---------------------------------------------------------------------------

def find_nmr_binary():
    """Locate the NMR binary."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', '..', 'nmr', 'build', 'nmr-server'),
        os.path.join(script_dir, '..', 'nmr', 'build', 'nmr-server'),
        os.environ.get('NMR_BIN', ''),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def find_parasolid_dyld():
    """Locate the Parasolid shared libraries for DYLD_LIBRARY_PATH."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', '..', 'nmr', 'scripts', '_deps',
                     'parasolid_37_1_180', 'parasolid', 'arm_macos', 'base', 'shared_object'),
        os.path.join(script_dir, '..', 'nmr', 'scripts', '_deps',
                     'parasolid_37_1_180', 'parasolid', 'arm_macos', 'base', 'shared_object'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return None


def convert_one(xt_path, npz_path, nmr_bin, env, tess_max_width):
    """Convert a single x_t file to NPZ via NMR."""
    cmd = [nmr_bin, '--convert', xt_path,
           '--tess-max-width', str(tess_max_width), '-o', npz_path]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip()
    return npz_path, None


def run_convert_step(metadata, output_dir, tess_max_width):
    """
    Step 1: Convert x_t files to NPZ via NMR.

    Reads raw/{sha256}.x_t, writes converted/{sha256}.npz.
    Skips instances that already have a converted NPZ.
    """
    converted_dir = os.path.join(output_dir, 'converted')
    os.makedirs(converted_dir, exist_ok=True)

    nmr_bin = find_nmr_binary()
    if not nmr_bin:
        print('WARNING: NMR binary not found (set NMR_BIN env var). Skipping conversion.')
        return []

    env = dict(os.environ)
    dyld = find_parasolid_dyld()
    if dyld:
        env['DYLD_LIBRARY_PATH'] = dyld

    to_convert = []
    already_done = []
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        npz_path = os.path.join(converted_dir, f'{sha256}.npz')
        if os.path.exists(npz_path):
            already_done.append({'sha256': sha256, 'has_converted': True})
            continue

        # Find the x_t source
        raw_xt = os.path.join(output_dir, 'raw', f'{sha256}.x_t')
        if not os.path.exists(raw_xt):
            source_path = row.get('source_path', '')
            if source_path and os.path.exists(source_path):
                raw_xt = source_path
            else:
                print(f'  No x_t file for {sha256}')
                continue

        to_convert.append((sha256, raw_xt, npz_path))

    if not to_convert:
        print(f'Convert: {len(already_done)} already done, 0 to convert')
        return already_done

    print(f'Convert: {len(already_done)} already done, {len(to_convert)} to convert '
          f'(--tess-max-width {tess_max_width})')

    records = list(already_done)
    n_ok = 0
    n_fail = 0
    for sha256, xt_path, npz_path in tqdm(to_convert, desc='Converting x_t → NPZ'):
        result_path, err = convert_one(xt_path, npz_path, nmr_bin, env, tess_max_width)
        if result_path:
            records.append({'sha256': sha256, 'has_converted': True})
            n_ok += 1
        else:
            print(f'  Convert failed for {sha256}: {err}')
            n_fail += 1

    print(f'  Converted {n_ok}, failed {n_fail}')
    return records


# ---------------------------------------------------------------------------
# Step 2: UDF computation (NPZ → mesh + UDF)
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
    vertices = data['vertices']
    faces = data['faces']
    edge_vertices = data['edge_vertices']

    # Warn if edge vertex ratio is suspiciously high (coarse tessellation)
    n_unique = len(set(edge_vertices.tolist())) if len(edge_vertices) > 0 else 0
    n_verts = vertices.shape[0]
    if n_verts > 0 and n_unique / n_verts > 0.5:
        print(f"  WARNING: {n_unique}/{n_verts} ({100*n_unique/n_verts:.0f}%) vertices "
              f"are edge vertices — tessellation is too coarse.")

    return vertices, faces, edge_vertices


def normalize_mesh(vertices, faces):
    """Normalize mesh to [-0.5, 0.5]^3 (TRELLIS convention)."""
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
    """
    import potpourri3d as pp3d

    if len(edge_vertex_indices) == 0:
        print("  Warning: no B-Rep edge vertices found, returning zero UDF")
        return np.zeros(len(vertices), dtype=np.float32)

    source_verts = np.array(sorted(set(edge_vertex_indices.tolist())), dtype=np.int32)
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    dist = solver.compute_distance_multisource(source_verts)
    return np.maximum(dist, 0.0).astype(np.float32)


def sample_surface_with_udf(vertices, faces, vertex_udf,
                             n_samples, edge_bias=0.5, edge_threshold=2.0):
    """
    Sample points on the mesh surface with interpolated UDF values.
    Uses edge-biased sampling so that near-edge regions get denser supervision.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area <= 0:
        total_area = 1.0

    face_mean_udf = vertex_udf[faces].mean(axis=1)
    near_edge_mask = face_mean_udf < edge_threshold
    n_near_edge = near_edge_mask.sum()

    if n_near_edge > 0 and edge_bias > 0:
        n_edge_samples    = int(n_samples * edge_bias)
        n_uniform_samples = n_samples - n_edge_samples

        edge_areas = areas.copy()
        edge_areas[~near_edge_mask] = 0.0
        edge_probs = edge_areas / edge_areas.sum()
        edge_tri_indices = np.random.choice(len(faces), size=n_edge_samples, p=edge_probs)

        uniform_probs = areas / total_area
        uniform_tri_indices = np.random.choice(len(faces), size=n_uniform_samples,
                                               p=uniform_probs)

        tri_indices = np.concatenate([edge_tri_indices, uniform_tri_indices])
    else:
        uniform_probs = areas / total_area
        tri_indices = np.random.choice(len(faces), size=n_samples, p=uniform_probs)

    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    bary = np.stack([u, v, w], axis=1)

    tri_verts = vertices[faces[tri_indices]]
    points = np.einsum('ijk,ij->ik', tri_verts, bary)

    corner_udf = vertex_udf[faces[tri_indices]]
    udf = np.einsum('ij,ij->i', corner_udf, bary)

    perm = np.random.permutation(n_samples)
    return points[perm].astype(np.float32), udf[perm].astype(np.float32)


def _process_udf(npz_path, sha256, output_dir,
                  num_samples, resolution=256,
                  edge_bias=0.5, edge_threshold=2.0):
    """Compute UDF for a single converted NPZ."""
    import trimesh

    mesh_path = os.path.join(output_dir, 'meshes', f'{sha256}.obj')
    udf_path  = os.path.join(output_dir, 'udf',    f'{sha256}.npz')

    if os.path.exists(mesh_path) and os.path.exists(udf_path):
        return {'sha256': sha256, 'has_mesh': True, 'has_udf': True}

    try:
        vertices, faces, edge_vertices = load_parasolid_npz(npz_path)

        if len(faces) == 0:
            print(f"  Warning: no faces in {sha256}")
            return None

        norm_verts, scale, offset = normalize_mesh(vertices, faces)

        mesh_obj = trimesh.Trimesh(vertices=norm_verts, faces=faces, process=False)
        mesh_obj.export(mesh_path)

        vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)

        voxel_size = 1.0 / resolution
        norm_udf = vertex_udf / voxel_size

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


def _udf_worker(args):
    return _process_udf(
        npz_path=args['npz_path'],
        sha256=args['sha256'],
        output_dir=args['output_dir'],
        num_samples=args['num_samples'],
        resolution=args['resolution'],
        edge_bias=args['edge_bias'],
        edge_threshold=args['edge_threshold'],
    )


def run_udf_step(metadata, output_dir, num_samples, resolution,
                  edge_bias, edge_threshold, max_workers):
    """
    Step 2: Compute UDF from converted NPZs.

    Reads converted/{sha256}.npz, writes meshes/{sha256}.obj + udf/{sha256}.npz.
    Skips instances that already have mesh + UDF.
    """
    converted_dir = os.path.join(output_dir, 'converted')
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'udf'),    exist_ok=True)

    work_items = []
    already_done = []
    skipped = 0
    for _, row in metadata.iterrows():
        sha256 = row['sha256']

        mesh_path = os.path.join(output_dir, 'meshes', f'{sha256}.obj')
        udf_path  = os.path.join(output_dir, 'udf',    f'{sha256}.npz')
        if os.path.exists(mesh_path) and os.path.exists(udf_path):
            already_done.append({'sha256': sha256, 'has_mesh': True, 'has_udf': True})
            continue

        npz_path = os.path.join(converted_dir, f'{sha256}.npz')
        if not os.path.exists(npz_path):
            skipped += 1
            continue

        work_items.append({
            'npz_path':     npz_path,
            'sha256':       sha256,
            'output_dir':   output_dir,
            'num_samples':  num_samples,
            'resolution':   resolution,
            'edge_bias':    edge_bias,
            'edge_threshold': edge_threshold,
        })

    print(f'UDF: {len(already_done)} already done, {len(work_items)} to process'
          + (f', {skipped} skipped (no converted NPZ)' if skipped else ''))

    if not work_items:
        return already_done

    n_workers = min(max_workers, len(work_items))
    processed = []
    if n_workers <= 1:
        for item in tqdm(work_items, desc='Computing UDF'):
            rec = _udf_worker(item)
            if rec is not None:
                processed.append(rec)
    else:
        with Pool(processes=n_workers) as pool:
            for rec in tqdm(pool.imap_unordered(_udf_worker, work_items),
                            total=len(work_items), desc='Computing UDF'):
                if rec is not None:
                    processed.append(rec)

    return already_done + processed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--tess_max_width', type=float, default=0.02,
                        help='Max facet width for NMR tessellation (default 0.02)')
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

    # Step 1: Convert x_t → NPZ
    convert_records = run_convert_step(metadata, opt.output_dir, opt.tess_max_width)

    # Step 2: Compute UDF from converted NPZs
    udf_records = run_udf_step(
        metadata, opt.output_dir,
        num_samples=opt.num_samples, resolution=opt.resolution,
        edge_bias=opt.edge_bias, edge_threshold=opt.edge_threshold,
        max_workers=opt.max_workers)

    # Save progress CSVs for build_metadata.py to merge
    if convert_records:
        pd.DataFrame.from_records(convert_records).to_csv(
            os.path.join(opt.output_dir, f'converted_{opt.rank}.csv'), index=False)
    if udf_records:
        pd.DataFrame.from_records(udf_records).to_csv(
            os.path.join(opt.output_dir, f'step_processed_{opt.rank}.csv'), index=False)

    print(f'\nDone. Run build_metadata.py to update metadata.csv')
