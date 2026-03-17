#!/usr/bin/env python3
"""
Prepare STEP files for TRELLIS training.

This script processes STEP files to generate:
1. Meshed OBJ files (for voxelization and feature extraction)
2. UDF ground truth (surface points + geodesic distance to BREP edges)

Usage:
    python prep_step_dataset.py STEPFiles --source_dir /path/to/step_files --output_dir datasets/STEPFiles
    
This should be run BEFORE the standard pipeline (render, voxelize, etc.)
"""

import os
import sys
import importlib
import argparse
import copy
import signal
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

# Add parent to path for udf_sample imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_step_and_mesh(filename, density_divisor=100.0):
    """
    Load STEP file, mesh it, and extract B-Rep edges.
    
    Returns:
        mesh: trimesh.Trimesh object
        edges: list of (i, j) vertex index pairs for B-Rep edges
        bbox_diagonal: diagonal length of the bounding box
    """
    import gmsh
    import trimesh
    
    gmsh.initialize()
    edges = []
    mesh = None
    bbox_diagonal = 1.0
    
    try:
        gmsh.open(filename)
        
        # Get bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag_vec = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        bbox_diagonal = np.linalg.norm(diag_vec)
        
        if bbox_diagonal == 0:
            target_edge_length = 1.0
        else:
            target_edge_length = bbox_diagonal / density_divisor
        
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_edge_length)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_edge_length)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        
        gmsh.model.occ.healShapes()
        gmsh.model.occ.synchronize()

        # Try meshing with fallback algorithms
        for algo in [6, 1, 5]:  # Frontal-Delaunay, MeshAdapt, Delaunay
            try:
                gmsh.option.setNumber("Mesh.Algorithm", algo)
                gmsh.model.mesh.generate(2)
                break
            except:
                gmsh.model.mesh.clear()
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_map = {tag: i for i, tag in enumerate(node_tags)}
        
        # Get triangle faces
        try:
            _, tri_node_tags = gmsh.model.mesh.getElementsByType(2)
            raw_faces = np.array(tri_node_tags, dtype=np.int32).reshape(-1, 3)
            faces = np.array([[tag_map[tag] for tag in face] for face in raw_faces], dtype=np.int32)
        except:
            faces = np.array([], dtype=np.int32).reshape(0, 3)
        
        # Get B-Rep edges (line elements)
        try:
            _, line_node_tags = gmsh.model.mesh.getElementsByType(1)
            raw_lines = np.array(line_node_tags, dtype=np.int32).reshape(-1, 2)
            edges = [tuple(sorted((tag_map[row[0]], tag_map[row[1]]))) for row in raw_lines]
        except:
            edges = []
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
    finally:
        gmsh.finalize()
    
    return mesh, edges, bbox_diagonal


def clean_mesh_and_remap_edges(mesh, edges):
    """Clean unreferenced vertices and update edge indices."""
    unique_face_indices = np.unique(mesh.faces)
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[unique_face_indices] = np.arange(len(unique_face_indices))
    
    import trimesh
    new_vertices = mesh.vertices[unique_face_indices]
    new_faces = old_to_new[mesh.faces]
    cleaned_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    new_edges = []
    for u, v in edges:
        if old_to_new[u] != -1 and old_to_new[v] != -1:
            new_edges.append((old_to_new[u], old_to_new[v]))
    
    return cleaned_mesh, new_edges


def normalize_mesh(mesh):
    """
    Normalize mesh to [-0.5, 0.5]^3 (TRELLIS convention).
    
    Returns:
        normalized_mesh: trimesh.Trimesh in normalized coords
        scale: scale factor used
        offset: offset (centroid) used
    """
    import trimesh
    
    vertices = mesh.vertices.copy()
    
    # Center at origin
    centroid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices -= centroid
    
    # Scale to fit in [-0.5, 0.5]^3
    max_extent = np.abs(vertices).max()
    if max_extent > 0:
        scale = 0.5 / max_extent * 0.95  # Leave small margin
    else:
        scale = 1.0
    vertices *= scale
    
    normalized_mesh = trimesh.Trimesh(
        vertices=vertices, 
        faces=mesh.faces.copy(),
        process=False
    )
    
    return normalized_mesh, scale, centroid


def compute_geodesic_udf(mesh, edges):
    """
    Compute geodesic distance from each vertex to nearest B-Rep edge vertex.
    
    Uses heat method for geodesic distance computation.
    """
    import potpourri3d as pp3d
    
    # Collect edge vertices
    edge_vertices = set()
    for i, j in edges:
        edge_vertices.update({i, j})
    
    if len(edge_vertices) == 0:
        print("Warning: No edge vertices found, returning zeros")
        return np.zeros(len(mesh.vertices), dtype=np.float32)
    
    source_verts = np.array(sorted(edge_vertices), dtype=np.int32)
    
    # Compute multi-source geodesic distance
    solver = pp3d.MeshHeatMethodDistanceSolver(mesh.vertices, mesh.faces)
    dist_to_edge = solver.compute_distance_multisource(source_verts)
    
    # Clamp to >= 0: the heat method can produce small negative artifacts
    # at/near source vertices (Poisson solve numerical error). True distance
    # at those vertices is exactly 0.
    dist_to_edge = np.maximum(dist_to_edge, 0.0)
    
    return dist_to_edge.astype(np.float32)


def sample_surface_with_udf(mesh, vertex_udf, n_samples, edge_bias=0.5, edge_threshold=2.0):
    """
    Sample random points on mesh surface with interpolated UDF values.
    
    Uses edge-biased sampling: a fraction of points are preferentially sampled
    from triangles near BREP edges (low UDF), giving denser supervision in the
    critical near-edge region.
    
    Args:
        mesh: trimesh.Trimesh
        vertex_udf: per-vertex UDF values (normalized by voxel size)
        n_samples: total number of points to sample
        edge_bias: fraction of samples drawn from near-edge triangles (0-1)
        edge_threshold: UDF threshold for "near-edge" (in voxel widths).
            Triangles where the mean vertex UDF < this are considered near-edge.
        
    Returns:
        points: [N, 3] surface points
        udf: [N] UDF values at those points
    """
    areas = mesh.area_faces
    
    # Compute per-face mean UDF (average of 3 corner UDF values)
    face_mean_udf = vertex_udf[mesh.faces].mean(axis=1)  # [F]
    
    # Split into near-edge and uniform sampling
    near_edge_mask = face_mean_udf < edge_threshold
    n_near_edge = near_edge_mask.sum()
    
    if n_near_edge > 0 and edge_bias > 0:
        n_edge_samples = int(n_samples * edge_bias)
        n_uniform_samples = n_samples - n_edge_samples
        
        # Near-edge sampling: area-weighted among near-edge faces only
        edge_areas = areas.copy()
        edge_areas[~near_edge_mask] = 0
        edge_probs = edge_areas / edge_areas.sum()
        edge_tri_indices = np.random.choice(len(mesh.faces), size=n_edge_samples, p=edge_probs)
        
        # Uniform sampling: area-weighted over ALL faces
        uniform_probs = areas / areas.sum()
        uniform_tri_indices = np.random.choice(len(mesh.faces), size=n_uniform_samples, p=uniform_probs)
        
        tri_indices = np.concatenate([edge_tri_indices, uniform_tri_indices])
        print(f"  Edge-biased sampling: {n_near_edge}/{len(mesh.faces)} near-edge faces "
              f"({n_edge_samples} edge + {n_uniform_samples} uniform = {n_samples} total)")
    else:
        # Fallback: pure area-weighted uniform sampling
        probs = areas / areas.sum()
        tri_indices = np.random.choice(len(mesh.faces), size=n_samples, p=probs)
        if n_near_edge == 0:
            print(f"  Warning: No near-edge faces (threshold={edge_threshold}), using uniform sampling")
    
    # Sample barycentric coordinates
    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    barycentrics = np.stack([u, v, w], axis=1)  # [N, 3]
    
    # Compute 3D positions
    tri_verts = mesh.vertices[mesh.faces[tri_indices]]  # [N, 3, 3]
    points = np.einsum('ijk,ij->ik', tri_verts, barycentrics)  # [N, 3]
    
    # Interpolate UDF values
    corner_udf = vertex_udf[mesh.faces[tri_indices]]  # [N, 3]
    udf = np.einsum('ij,ij->i', corner_udf, barycentrics)  # [N]
    
    # Shuffle so edge/uniform samples are interleaved
    perm = np.random.permutation(n_samples)
    
    return points[perm].astype(np.float32), udf[perm].astype(np.float32)


def _process_step(file_path, sha256, output_dir, num_samples, density, resolution=256, edge_bias=0.5, edge_threshold=2.0):
    """Process a single STEP file."""
    import trimesh
    
    mesh_dir = os.path.join(output_dir, 'meshes')
    udf_dir = os.path.join(output_dir, 'udf')
    
    mesh_path = os.path.join(mesh_dir, f'{sha256}.obj')
    udf_path = os.path.join(udf_dir, f'{sha256}.npz')
    
    # Skip if already processed
    if os.path.exists(mesh_path) and os.path.exists(udf_path):
        return {'sha256': sha256, 'has_mesh': True, 'has_udf': True}
    
    try:
        # Load and mesh STEP file
        raw_mesh, raw_edges, bbox_diag = load_step_and_mesh(file_path, density)
        
        if len(raw_mesh.faces) == 0:
            print(f"Warning: No faces generated for {sha256}")
            return None
        
        # Clean mesh
        mesh, edges = clean_mesh_and_remap_edges(raw_mesh, raw_edges)
        
        if len(edges) == 0:
            print(f"Warning: No B-Rep edges found for {sha256}")
        
        # Normalize to [-0.5, 0.5]^3
        normalized_mesh, scale, offset = normalize_mesh(mesh)
        
        # Save OBJ mesh
        normalized_mesh.export(mesh_path)
        
        # Compute geodesic UDF on vertices
        vertex_udf = compute_geodesic_udf(normalized_mesh, edges)
        
        # Normalize UDF by voxel size so UDF=1.0 means "one voxel width away from edge"
        voxel_size = 1.0 / resolution
        normalized_udf = vertex_udf / voxel_size
        
        # Sample surface points with UDF
        points, udf = sample_surface_with_udf(normalized_mesh, normalized_udf, num_samples,
                                               edge_bias=edge_bias, edge_threshold=edge_threshold)
        
        # Save UDF data
        np.savez(
            udf_path,
            points=points,
            udf=udf,
            # Also save metadata for reference
            scale=np.array([scale], dtype=np.float32),
            offset=offset.astype(np.float32),
            num_edges=np.array([len(edges)], dtype=np.int32),
        )
        
        return {
            'sha256': sha256,
            'has_mesh': True,
            'has_udf': True,
            'num_faces': len(normalized_mesh.faces),
            'num_vertices': len(normalized_mesh.vertices),
            'num_edges': len(edges),
        }
        
    except Exception as e:
        print(f"Error processing {sha256}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _worker_init():
    """Initializer for each worker process — ignore SIGINT so the parent handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_fn(args):
    """
    Top-level worker function for multiprocessing (must be picklable).
    Receives a single dict with all arguments.
    """
    return _process_step(
        file_path=args['file_path'],
        sha256=args['sha256'],
        output_dir=args['output_dir'],
        num_samples=args['num_samples'],
        density=args['density'],
        resolution=args['resolution'],
        edge_bias=args['edge_bias'],
        edge_threshold=args['edge_threshold'],
    )


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of surface points to sample for UDF')
    parser.add_argument('--density', type=float, default=100.0,
                        help='Mesh density (edges per diagonal)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Voxel grid resolution for UDF normalization (default: 256)')
    parser.add_argument('--edge_bias', type=float, default=0.5,
                        help='Fraction of samples biased toward near-edge triangles (0=uniform, 1=all edge)')
    parser.add_argument('--edge_threshold', type=float, default=2.0,
                        help='UDF threshold (in voxel widths) for near-edge triangles')
    parser.add_argument('--instances', type=str, default=None,
                        help='Specific instances to process (comma-separated or file)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-model timeout in seconds (default 300 = 5 min, 0 = no timeout)')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel worker processes (default 4). '
                             'Each gets its own gmsh instance. Set 1 for sequential.')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create output directories
    os.makedirs(os.path.join(opt.output_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'udf'), exist_ok=True)


    # Build or load metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        print('Building metadata from source directory...')
        metadata = dataset_utils.get_metadata(**opt)
        metadata.to_csv(os.path.join(opt.output_dir, 'metadata.csv'), index=False)
        print(f'Created metadata.csv with {len(metadata)} instances')
        
        # Also download/symlink files
        print('Linking source files...')
        downloaded = dataset_utils.download(metadata, opt.output_dir)
        print(f'Linked {len(downloaded)} files')
        
        # Reload metadata with updated paths
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter instances if specified
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    
    # Shard for distributed processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
    # Filter already processed
    records = []
    for sha256 in copy.copy(metadata['sha256'].values):
        mesh_path = os.path.join(opt.output_dir, 'meshes', f'{sha256}.obj')
        udf_path = os.path.join(opt.output_dir, 'udf', f'{sha256}.npz')
        if os.path.exists(mesh_path) and os.path.exists(udf_path):
            records.append({'sha256': sha256, 'has_mesh': True, 'has_udf': True})
            metadata = metadata[metadata['sha256'] != sha256]
    
    # Build work items
    work_items = []
    skipped_missing = 0
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        local_path = row.get('local_path')
        if local_path:
            file_path = os.path.join(opt.output_dir, local_path)
        else:
            file_path = row.get('source_path')
        
        if file_path and os.path.exists(file_path):
            work_items.append({
                'file_path': file_path,
                'sha256': sha256,
                'output_dir': opt.output_dir,
                'num_samples': opt.num_samples,
                'density': opt.density,
                'resolution': opt.resolution,
                'edge_bias': opt.edge_bias,
                'edge_threshold': opt.edge_threshold,
            })
        else:
            print(f"File not found: {file_path}")
            skipped_missing += 1

    n_workers = min(opt.max_workers, len(work_items)) if work_items else 1
    timeout = opt.timeout if opt.timeout > 0 else None
    print(f'Processing {len(work_items)} STEP files with {n_workers} workers '
          f'(timeout={timeout}s)...')
    if skipped_missing:
        print(f'  Skipped {skipped_missing} missing files')

    processed_records = []
    n_timeout = 0
    n_error = 0

    if n_workers <= 1:
        # Sequential fallback (useful for debugging)
        for item in tqdm(work_items, desc='Processing STEP files'):
            record = _worker_fn(item)
            if record is not None:
                processed_records.append(record)
    else:
        # Multiprocess: each worker gets its own gmsh instance
        pool = Pool(processes=n_workers, initializer=_worker_init)
        try:
            async_results = []
            for item in work_items:
                ar = pool.apply_async(_worker_fn, (item,))
                async_results.append((item['sha256'], ar))
            
            for sha256, ar in tqdm(async_results, desc='Processing STEP files'):
                try:
                    record = ar.get(timeout=timeout)
                    if record is not None:
                        processed_records.append(record)
                except KeyboardInterrupt:
                    print('\nInterrupted by user. Saving progress...')
                    pool.terminate()
                    break
                except Exception as e:
                    if 'TimeoutError' in type(e).__name__ or isinstance(e, TimeoutError):
                        print(f'  TIMEOUT: {sha256} exceeded {timeout}s — skipping')
                        n_timeout += 1
                    else:
                        print(f'  ERROR: {sha256}: {e}')
                        n_error += 1
            
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print('\nInterrupted. Terminating workers...')
            pool.terminate()
            pool.join()

    # Combine and save records
    processed = pd.DataFrame.from_records(processed_records + records)
    processed.to_csv(os.path.join(opt.output_dir, f'step_processed_{opt.rank}.csv'), index=False)
    
    n_success = len(processed_records)
    n_skipped = len(records)
    print(f'\nDone: {n_success} processed, {n_skipped} already existed, '
          f'{n_timeout} timed out, {n_error} errors')
    print(f'Total in output: {len(processed)}')
    print('Run build_metadata.py to update metadata.csv')
