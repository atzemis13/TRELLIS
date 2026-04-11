#!/usr/bin/env python3
"""
Test that UDF computation produces near-zero values at B-Rep edge vertices.

Background:
    The heat method (potpourri3d.MeshHeatMethodDistanceSolver) computes geodesic
    distance from source vertices (B-Rep edge vertices) to all mesh vertices.
    The distance at source vertices should be exactly zero, but in practice the
    heat method produces significant errors — up to 3-5 voxel widths — at and
    near source vertices. This is especially bad when many source vertices are
    densely packed along B-Rep edges with thin/elongated triangles.

    This corrupts the UDF ground truth: the model trains on data where points
    ON B-Rep edges have non-zero UDF, making it impossible to learn sharp edge
    detection.

    The test loads real NMR-converted meshes and verifies that whatever UDF
    computation method is used, edge vertices get near-zero values.

Usage:
    pytest tests/test_udf_at_edges.py -v
    python tests/test_udf_at_edges.py  # standalone
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_toolkits'))
from prep_parasolid_dataset import compute_geodesic_udf, normalize_mesh


CONVERTED_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'overfit5', 'converted')

PARTS = {
    'Mug': 'ee88ce4747f66c9d0c97c546ad0b920c2029f9b3dc6bd5d59591890561c7aefe',
    'Block': '914dc29aa0451d3dbd2dd687f436528e739ac62d736f881d0bf69fefee1755f8',
    'DoorStop': '74d7bab4de82def1adf840f1dc908388ddfef8f14dbf99c6b00d822c8124ac55',
    'BikeClamp': '51112d9fdb2136309e571ee066b0dacf8bc93edec75c3ac5d44e6ebc91c0d1c9',
    'Flange': '3b663a36a2a3d92068d1a528fa830f42cdcefff216e8b551f14df84988249b5c',
}

VOXEL_SIZE = 1.0 / 256


def load_part(name):
    sha = PARTS[name]
    path = os.path.join(CONVERTED_DIR, f'{sha}.npz')
    if not os.path.exists(path):
        pytest.skip(f'Converted NPZ not found: {path}')
    data = np.load(path)
    verts = data['vertices']
    faces = data['faces']
    edge_vertices = data['edge_vertices']
    norm_verts, _, _ = normalize_mesh(verts, faces)
    return norm_verts, faces, edge_vertices


@pytest.mark.parametrize('part_name', list(PARTS.keys()))
def test_edge_vertices_have_near_zero_udf(part_name):
    """
    Edge vertices (B-Rep edge sources) must have UDF < 0.1 voxel widths.

    This is the fundamental requirement: vertices that ARE on B-Rep edges
    must have distance-to-nearest-edge ≈ 0. Currently the heat method fails
    this for ~50% of edge vertices.
    """
    norm_verts, faces, edge_vertices = load_part(part_name)
    unique_ev = sorted(set(edge_vertices.tolist()))

    vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)
    normalized_udf = vertex_udf / VOXEL_SIZE

    ev_udf = normalized_udf[unique_ev]
    max_edge_udf = ev_udf.max()
    pct_above_threshold = 100 * (ev_udf > 0.1).mean()

    print(f'\n{part_name}: edge vertex UDF max={max_edge_udf:.4f}, '
          f'{pct_above_threshold:.1f}% above 0.1')

    # All edge vertices must have UDF < 0.1 voxel widths
    assert max_edge_udf < 0.1, (
        f'{part_name}: {pct_above_threshold:.1f}% of edge vertices have UDF > 0.1 '
        f'(max={max_edge_udf:.4f}). The UDF computation is not accurate at edge sources.'
    )


@pytest.mark.parametrize('part_name', list(PARTS.keys()))
def test_near_edge_faces_have_low_udf(part_name):
    """
    Vertices on faces adjacent to B-Rep edges must have a smooth gradient,
    not a sharp spike. Specifically, vertices that share a face with an edge
    vertex should have UDF < 2.0 voxel widths.
    """
    norm_verts, faces, edge_vertices = load_part(part_name)
    unique_ev = set(edge_vertices.tolist())

    vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)
    normalized_udf = vertex_udf / VOXEL_SIZE

    # Find vertices adjacent to edge vertices (1-ring neighbors)
    neighbors = set()
    for face in faces:
        face_verts = set(int(v) for v in face)
        if face_verts & unique_ev:
            neighbors.update(face_verts - unique_ev)

    if not neighbors:
        pytest.skip(f'{part_name}: no non-edge neighbors found')

    neighbor_udf = normalized_udf[sorted(neighbors)]
    max_neighbor_udf = neighbor_udf.max()
    mean_neighbor_udf = neighbor_udf.mean()

    print(f'\n{part_name}: adjacent vertex UDF mean={mean_neighbor_udf:.4f}, '
          f'max={max_neighbor_udf:.4f}')

    # Adjacent vertices should have reasonable UDF (within a few voxel widths)
    assert mean_neighbor_udf < 2.0, (
        f'{part_name}: mean UDF of edge-adjacent vertices is {mean_neighbor_udf:.4f}, '
        f'expected < 2.0. The UDF gradient near edges is not smooth.'
    )


@pytest.mark.parametrize('part_name', list(PARTS.keys()))
def test_sampled_edge_points_have_near_zero_udf(part_name):
    """
    Points sampled along B-Rep edge segments (between edge vertex pairs)
    must have near-zero UDF. This tests that the UDF is correct not just
    at edge vertices but along the full edge curves.
    """
    norm_verts, faces, edge_vertices = load_part(part_name)
    unique_ev = set(edge_vertices.tolist())

    vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)
    normalized_udf = vertex_udf / VOXEL_SIZE

    # Find B-Rep edge segments (mesh edges where both endpoints are edge verts)
    brep_edges = set()
    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            if a in unique_ev and b in unique_ev:
                brep_edges.add((min(a, b), max(a, b)))

    # Sample midpoints of edge segments, interpolate UDF
    midpoint_udfs = []
    for a, b in brep_edges:
        mid_udf = (normalized_udf[a] + normalized_udf[b]) / 2.0
        midpoint_udfs.append(mid_udf)

    midpoint_udfs = np.array(midpoint_udfs)
    max_mid_udf = midpoint_udfs.max()
    pct_above = 100 * (midpoint_udfs > 0.2).mean()

    print(f'\n{part_name}: {len(brep_edges)} edge segments, '
          f'midpoint UDF max={max_mid_udf:.4f}, {pct_above:.1f}% above 0.2')

    # Edge segment midpoints should have very low UDF
    assert max_mid_udf < 0.2, (
        f'{part_name}: {pct_above:.1f}% of edge segment midpoints have UDF > 0.2 '
        f'(max={max_mid_udf:.4f}). UDF is inaccurate along B-Rep edges.'
    )


if __name__ == '__main__':
    # Run standalone with verbose output
    for name in PARTS:
        print(f'\n{"="*60}')
        print(f'Testing {name}')
        print(f'{"="*60}')
        try:
            norm_verts, faces, edge_vertices = load_part(name)
        except Exception as e:
            print(f'  SKIP: {e}')
            continue

        unique_ev = sorted(set(edge_vertices.tolist()))
        vertex_udf = compute_geodesic_udf(norm_verts, faces, edge_vertices)
        normalized_udf = vertex_udf / VOXEL_SIZE
        ev_udf = normalized_udf[unique_ev]

        print(f'  Edge vertices: {len(unique_ev)} / {len(norm_verts)}')
        print(f'  Edge vert UDF: min={ev_udf.min():.4f} max={ev_udf.max():.4f} '
              f'mean={ev_udf.mean():.4f}')
        print(f'  UDF > 0.1: {(ev_udf > 0.1).sum()} ({100*(ev_udf > 0.1).mean():.1f}%)')
        print(f'  UDF > 0.01: {(ev_udf > 0.01).sum()} ({100*(ev_udf > 0.01).mean():.1f}%)')

        status = 'PASS' if ev_udf.max() < 0.1 else 'FAIL'
        print(f'  Result: {status}')
