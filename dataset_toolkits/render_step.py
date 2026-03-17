#!/usr/bin/env python3
"""
Render multiview images from STEP files using step_renderer.

This creates renders in the format expected by TRELLIS feature extraction,
including the transforms.json file with camera matrices.

Usage:
    python render_step.py STEPFiles --output_dir datasets/STEPFiles [--num_views 150]

Prerequisites:
    - step_renderer must be built: cd step_renderer && mkdir build && cd build && cmake .. && make
    - STEP files must be preprocessed: python prep_step_dataset.py STEPFiles --source_dir ... --output_dir ...
"""

import os
import sys
import json
import copy
import importlib
import argparse
import subprocess
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Halton/Hammersley sequences for camera sampling (same as TRELLIS)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[d], n) for d in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return float(phi), float(theta)


def compute_transform_matrix(yaw, pitch, radius):
    """
    Compute 4x4 camera-to-world transform matrix.
    
    Camera looks at origin from position on a sphere.
    Convention: Z-up world, camera looks along -Z in camera space.
    """
    # Camera position
    x = radius * np.cos(yaw) * np.cos(pitch)
    y = radius * np.sin(yaw) * np.cos(pitch)
    z = radius * np.sin(pitch)
    position = np.array([x, y, z])
    
    # Look at origin
    forward = -position / np.linalg.norm(position)  # Camera looks toward origin
    
    # Up vector (world Z)
    world_up = np.array([0, 0, 1])
    
    # Right vector
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Camera is looking straight up or down
        right = np.array([1, 0, 0])
    else:
        right = right / np.linalg.norm(right)
    
    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build 4x4 transformation matrix (camera-to-world)
    # Column vectors: right, up, -forward, position
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward  # OpenGL convention: camera looks along -Z
    transform[:3, 3] = position
    
    return transform.tolist()


def _render_step(file_path, sha256, output_dir, num_views, renderer_path):
    """Render a single STEP file."""
    
    output_folder = os.path.join(output_dir, 'renders', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if already rendered
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        with open(os.path.join(output_folder, 'transforms.json')) as f:
            data = json.load(f)
            if len(data.get('frames', [])) >= num_views:
                return {'sha256': sha256, 'rendered': True}
    
    # Generate camera views
    offset = (np.random.rand(), np.random.rand())
    fov = 40 / 180 * np.pi
    radius = 2.0
    
    views = []
    frames = []
    
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        views.append({
            "yaw": yaw,
            "pitch": pitch,
            "radius": radius,
            "fov": fov,
            "output": os.path.join(output_folder, f"{i:03d}.png")
        })
        
        frames.append({
            "file_path": f"{i:03d}.png",
            "camera_angle_x": fov,
            "transform_matrix": compute_transform_matrix(yaw, pitch, radius)
        })
    
    # Load normalization params from UDF data so the renderer applies the
    # same coordinate transform as the mesher (critical for UDF alignment).
    udf_path = os.path.join(output_dir, 'udf', f'{sha256}.npz')
    scale = 1.0
    offset_vec = [0, 0, 0]
    norm_args = []
    if os.path.exists(udf_path):
        udf_data = np.load(udf_path)
        if 'scale' in udf_data:
            scale = float(udf_data['scale'][0])
        if 'offset' in udf_data:
            offset_vec = udf_data['offset'].tolist()
        norm_args = [
            '--scale', str(scale),
            '--offset', str(offset_vec[0]), str(offset_vec[1]), str(offset_vec[2]),
        ]

    # Write views.json for step_renderer
    views_file = os.path.join(output_folder, "views.json")
    with open(views_file, "w") as f:
        json.dump(views, f)
    
    # Run step_renderer
    try:
        result = subprocess.run(
            [renderer_path, "--object", file_path, "--views", views_file] + norm_args,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Renderer failed for {sha256}: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Renderer not found at {renderer_path}")
        print("Build with: cd step_renderer && mkdir -p build && cd build && cmake .. && make")
        return None
    
    # Write transforms.json
    transforms = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": offset_vec,
        "frames": frames
    }
    
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Also copy/symlink the mesh for voxelization
    # (voxelize.py expects renders/{sha256}/mesh.ply but we have meshes/{sha256}.obj)
    mesh_obj = os.path.join(output_dir, 'meshes', f'{sha256}.obj')
    mesh_ply = os.path.join(output_folder, 'mesh.ply')
    
    if os.path.exists(mesh_obj) and not os.path.exists(mesh_ply):
        # Convert OBJ to PLY using trimesh
        try:
            import trimesh
            mesh = trimesh.load(mesh_obj)
            mesh.export(mesh_ply)
        except Exception as e:
            print(f"Could not convert mesh to PLY for {sha256}: {e}")
    
    return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory with processed STEP data')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    parser.add_argument('--renderer', type=str, default=None,
                        help='Path to step_render executable')
    parser.add_argument('--instances', type=str, default=None,
                        help='Specific instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=4)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Find renderer
    if opt.renderer is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'step_renderer', 'build', 'step_render'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'step_renderer', 'build', 'render_step'),
            '/usr/local/bin/step_render',
        ]
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                opt.renderer = c
                break
        
        if opt.renderer is None:
            print("Error: step_render executable not found")
            print("Build with: cd step_renderer && mkdir -p build && cd build && cmake .. && make")
            sys.exit(1)
    
    print(f"Using renderer: {opt.renderer}")

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)

    # Load metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found. Run prep_step_dataset.py first.')
    
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter to instances with meshes
    mesh_dir = os.path.join(opt.output_dir, 'meshes')
    has_mesh = metadata['sha256'].apply(
        lambda x: os.path.exists(os.path.join(mesh_dir, f'{x}.obj'))
    )
    metadata = metadata[has_mesh]
    
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    
    # Shard
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
    # Filter already rendered
    records = []
    for sha256 in copy.copy(metadata['sha256'].values):
        transforms_path = os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')
        if os.path.exists(transforms_path):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
    
    print(f'Rendering {len(metadata)} STEP files...')
    
    # Process
    func = partial(
        _render_step,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        renderer_path=opt.renderer,
    )
    
    rendered = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func,
        max_workers=opt.max_workers,
        desc='Rendering STEP files'
    )
    
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
    
    print(f'Rendered {len(rendered)} files')
    print('Run build_metadata.py to update metadata.csv')
