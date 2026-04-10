#!/usr/bin/env python3
"""
Decode mesh + UDF from trained models and visualize B-Rep edge detection.

Given SLAT latents for each part:
1. Decode mesh via pretrained mesh decoder
2. Decode UDF via trained UDF decoder
3. Interpolate UDF at mesh vertices
4. Paint near-zero UDF values as edges
5. Export PLY files and renders

Usage:
    python scripts/decode_and_visualize.py \
        --data_dir datasets/overfit5 \
        --udf_ckpt results/overfit5_udf/ckpts/decoder_ema0.9999_step0010000.pt \
        --udf_config configs/vae/overfit5_udf.json \
        --output_dir results/overfit5_udf/visualization \
        --thresholds 0.05 0.1 0.2 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import trellis.models as models
from trellis.modules.sparse import SparseTensor
from trellis.representations.mesh.udf import interpolate_udf_to_points


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Dataset directory (with latents/ and metadata.csv)')
    p.add_argument('--udf_ckpt', required=True, help='Path to trained UDF decoder checkpoint (.pt)')
    p.add_argument('--udf_config', default='configs/vae/overfit5_udf.json', help='UDF decoder config')
    p.add_argument('--mesh_model', default='microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16',
                   help='Pretrained mesh decoder (HuggingFace path or local)')
    p.add_argument('--output_dir', required=True, help='Output directory for visualizations')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5],
                   help='UDF thresholds for edge painting')
    p.add_argument('--instances', type=str, default=None,
                   help='Comma-separated SHA256s or path to file (default: all in dataset)')
    p.add_argument('--latent_model', default='dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16',
                   help='Latent model name (subdirectory in latents/)')
    return p.parse_args()


def load_udf_decoder(config_path, ckpt_path):
    """Load trained UDF decoder from .pt checkpoint."""
    with open(config_path) as f:
        cfg = json.load(f)
    decoder_cfg = cfg['models']['decoder']
    decoder = getattr(models, decoder_cfg['name'])(**decoder_cfg['args']).cuda()
    state_dict = torch.load(ckpt_path, map_location='cuda')
    decoder.load_state_dict(state_dict)
    decoder.eval()
    print(f'Loaded UDF decoder from {ckpt_path}')
    return decoder


def load_mesh_decoder(model_path):
    """Load pretrained mesh decoder."""
    decoder = models.from_pretrained(model_path).eval().cuda()
    print(f'Loaded mesh decoder from {model_path}')
    return decoder


def load_latent(data_dir, latent_model, sha256):
    """Load SLAT latent and construct SparseTensor."""
    path = os.path.join(data_dir, 'latents', latent_model, f'{sha256}.npz')
    data = np.load(path)
    coords = torch.from_numpy(data['coords']).int()
    feats = torch.from_numpy(data['feats']).float()
    n = coords.shape[0]
    # Add batch dimension
    coords = torch.cat([torch.zeros(n, 1, dtype=torch.int32), coords], dim=1)
    return SparseTensor(coords=coords, feats=feats).cuda()


def export_ply_with_colors(path, vertices, faces, colors):
    """
    Export mesh as PLY with per-vertex RGB colors.

    Args:
        vertices: [V, 3] float
        faces: [F, 3] int
        colors: [V, 3] uint8
    """
    v = vertices.shape[0]
    f = faces.shape[0]
    with open(path, 'w') as fp:
        fp.write('ply\n')
        fp.write('format ascii 1.0\n')
        fp.write(f'element vertex {v}\n')
        fp.write('property float x\n')
        fp.write('property float y\n')
        fp.write('property float z\n')
        fp.write('property uchar red\n')
        fp.write('property uchar green\n')
        fp.write('property uchar blue\n')
        fp.write(f'element face {f}\n')
        fp.write('property list uchar int vertex_indices\n')
        fp.write('end_header\n')
        for i in range(v):
            fp.write(f'{vertices[i,0]:.6f} {vertices[i,1]:.6f} {vertices[i,2]:.6f} '
                     f'{colors[i,0]} {colors[i,1]} {colors[i,2]}\n')
        for i in range(f):
            fp.write(f'3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n')


def paint_edges(vertex_udf, threshold):
    """
    Create vertex colors: red for near-edge (UDF < threshold), gray for surface.

    Returns [V, 3] uint8 array.
    """
    v = vertex_udf.shape[0]
    colors = np.full((v, 3), 180, dtype=np.uint8)  # gray base
    edge_mask = vertex_udf < threshold
    # Gradient: bright red at UDF=0, fading toward threshold
    t = np.clip(vertex_udf[edge_mask] / threshold, 0, 1)
    colors[edge_mask, 0] = 255  # R
    colors[edge_mask, 1] = (t * 180).astype(np.uint8)  # G: 0 at edge, 180 at threshold
    colors[edge_mask, 2] = (t * 180).astype(np.uint8)  # B: same
    return colors


def render_multiview(vertices, faces, vertex_udf, threshold, output_path):
    """Render 4-angle views of the painted mesh using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print('  matplotlib not available, skipping renders')
        return

    colors = paint_edges(vertex_udf, threshold)
    face_colors = colors[faces].mean(axis=1) / 255.0  # per-face average color

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': '3d'})
    elevations = [20, 20, 60, -20]
    azimuths = [0, 90, 45, 135]

    verts_np = vertices
    for ax, elev, azim in zip(axes, elevations, azimuths):
        tri_verts = verts_np[faces]
        poly = Poly3DCollection(tri_verts, alpha=0.9)
        poly.set_facecolor(face_colors)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

        lim = 0.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f'elev={elev} azim={azim}')

    fig.suptitle(f'UDF Edge Paint (threshold={threshold:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine instances
    if args.instances:
        if os.path.exists(args.instances):
            with open(args.instances) as f:
                instances = [l.strip() for l in f if l.strip()]
        else:
            instances = args.instances.split(',')
    else:
        # Find all latents in data_dir
        latent_dir = os.path.join(args.data_dir, 'latents', args.latent_model)
        instances = [f.replace('.npz', '') for f in os.listdir(latent_dir) if f.endswith('.npz')]

    print(f'Processing {len(instances)} instances')
    print(f'Thresholds: {args.thresholds}')

    # Load models
    mesh_decoder = load_mesh_decoder(args.mesh_model)
    udf_decoder = load_udf_decoder(args.udf_config, args.udf_ckpt)

    summary_rows = []

    for sha256 in instances:
        print(f'\n=== {sha256[:16]}... ===')

        # Load latent
        st = load_latent(args.data_dir, args.latent_model, sha256)
        print(f'  Latent: {st.feats.shape[0]} voxels, {st.feats.shape[1]} channels')

        with torch.no_grad():
            # Decode mesh
            mesh_results = mesh_decoder(st)
            mesh = mesh_results[0]
            if not mesh.success:
                print(f'  WARN: mesh extraction failed, skipping')
                continue
            verts = mesh.vertices.cpu().numpy()
            faces = mesh.faces.cpu().numpy()
            print(f'  Mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces')

            # Decode UDF
            udf_results = udf_decoder(st)
            udf_result = udf_results[0]
            if not udf_result.success:
                print(f'  WARN: UDF extraction failed, skipping')
                continue

            # Interpolate UDF at mesh vertices
            vertex_udf = interpolate_udf_to_points(
                udf_result.udf_grid, mesh.vertices, res=256
            ).cpu().numpy()
            print(f'  UDF range: [{vertex_udf.min():.4f}, {vertex_udf.max():.4f}], '
                  f'median={np.median(vertex_udf):.4f}')

        # Save raw data (for post-hoc threshold adjustment)
        raw_path = os.path.join(args.output_dir, f'{sha256}_raw.npz')
        np.savez(raw_path, vertices=verts, faces=faces, vertex_udf=vertex_udf)
        print(f'  Saved raw data: {raw_path}')

        # Export PLY files for each threshold
        for t in args.thresholds:
            n_edge = (vertex_udf < t).sum()
            pct = 100.0 * n_edge / len(vertex_udf)
            colors = paint_edges(vertex_udf, t)
            ply_path = os.path.join(args.output_dir, f'{sha256}_mesh_udf_t{t:.2f}.ply')
            export_ply_with_colors(ply_path, verts, faces, colors)
            print(f'  t={t:.2f}: {n_edge}/{len(vertex_udf)} edge verts ({pct:.1f}%) -> {ply_path}')

        # Render multi-angle images at default threshold
        default_t = 0.1
        render_path = os.path.join(args.output_dir, f'{sha256}_renders.png')
        render_multiview(verts, faces, vertex_udf, default_t, render_path)

        summary_rows.append({
            'sha256': sha256,
            'n_verts': verts.shape[0],
            'n_faces': faces.shape[0],
            'udf_min': float(vertex_udf.min()),
            'udf_max': float(vertex_udf.max()),
            'udf_median': float(np.median(vertex_udf)),
            **{f'edge_pct_t{t:.2f}': 100.0 * (vertex_udf < t).sum() / len(vertex_udf)
               for t in args.thresholds},
        })

    # Write summary HTML
    html_path = os.path.join(args.output_dir, 'summary.html')
    with open(html_path, 'w') as f:
        f.write('<html><head><style>body{font-family:monospace;} img{max-width:100%;} '
                'table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 8px;}</style></head><body>\n')
        f.write('<h1>UDF Edge Detection Results</h1>\n')
        f.write(f'<p>Thresholds: {args.thresholds}</p>\n')
        f.write(f'<p>UDF checkpoint: {args.udf_ckpt}</p>\n')

        f.write('<table><tr><th>Part</th><th>Verts</th><th>Faces</th>'
                '<th>UDF min</th><th>UDF median</th>')
        for t in args.thresholds:
            f.write(f'<th>Edge% t={t}</th>')
        f.write('</tr>\n')
        for row in summary_rows:
            f.write(f'<tr><td>{row["sha256"][:16]}...</td>'
                    f'<td>{row["n_verts"]}</td><td>{row["n_faces"]}</td>'
                    f'<td>{row["udf_min"]:.4f}</td><td>{row["udf_median"]:.4f}</td>')
            for t in args.thresholds:
                f.write(f'<td>{row[f"edge_pct_t{t:.2f}"]:.1f}%</td>')
            f.write('</tr>\n')
        f.write('</table>\n')

        for row in summary_rows:
            sha = row['sha256']
            f.write(f'<h2>{sha[:16]}...</h2>\n')
            render_file = f'{sha}_renders.png'
            if os.path.exists(os.path.join(args.output_dir, render_file)):
                f.write(f'<img src="{render_file}" />\n')

        f.write('</body></html>\n')
    print(f'\nSummary: {html_path}')


if __name__ == '__main__':
    main()
