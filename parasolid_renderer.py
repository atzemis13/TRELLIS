#!/usr/bin/env python3
"""
Parasolid renderer — wrapper around NMR's --snapshot CLI.

Uses NMR's built-in offscreen renderer (Qt OpenGL) which:
  - Loads .x_t files directly via Parasolid
  - Samples actual B-Rep curves (PK_CURVE_eval) for smooth edge lines
  - Renders with Phong shading, MSAA, geometry-shader thick edges
  - Auto-fits camera to bounding sphere (orthographic projection)

CLI (same interface as step_renderer):
    python parasolid_renderer.py \\
        --object   path/to/part.x_t \\
        --views    path/to/views.json \\
        [--output_dir /path/to/dataset/dir] \\
        [--width 512] [--height 512]

views.json format (TRELLIS convention):
    [{"yaw": float, "pitch": float, "radius": float, "fov": float,
      "output": "path/to/NNN.png"}, ...]

NMR binary lookup:
    1. ../nmr/build_qt/nmr   (Qt build, relative to this script)
    2. ../nmr/build/nmr       (if Qt build not available)
    3. NMR_BIN environment variable
"""

import os, sys, json, argparse, hashlib, shutil, subprocess, tempfile
import math

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--object',     required=True, help='Path to .x_t file')
    p.add_argument('--views',      required=True, help='Path to views.json')
    p.add_argument('--output_dir', default=None,
                   help='Dataset root (accepted for compatibility, not used)')
    p.add_argument('--scale',      type=float, default=None,
                   help='Ignored (NMR auto-fits to bounding sphere)')
    p.add_argument('--offset',     type=float, nargs=3, default=None,
                   help='Ignored (NMR auto-fits to bounding sphere)')
    p.add_argument('--width',  type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Locate NMR binary and Parasolid shared libraries
# ---------------------------------------------------------------------------

def _find_nmr():
    """Find the NMR binary (prefer Qt build with snapshot support)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'nmr', 'build_qt', 'nmr'),
        os.path.join(script_dir, '..', 'nmr', 'build', 'nmr'),
        os.environ.get('NMR_BIN', ''),
    ]
    for c in candidates:
        if c:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
    return None

def _nmr_dyld():
    """Find the Parasolid shared_object directory for DYLD_LIBRARY_PATH."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'nmr', 'scripts', '_deps',
                     'parasolid_37_1_180', 'parasolid', 'arm_macos',
                     'base', 'shared_object'),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c
    return ''

def _nmr_presets_file():
    """Find the default snapshot presets file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c = os.path.join(script_dir, '..', 'nmr', 'resources', 'snapshot_presets.json')
    c = os.path.abspath(c)
    return c if os.path.exists(c) else None

# ---------------------------------------------------------------------------
# Camera conversion: TRELLIS → NMR
# ---------------------------------------------------------------------------

def trellis_to_nmr_camera(yaw, pitch):
    """
    Convert TRELLIS spherical camera (yaw, pitch in radians) to NMR orbit
    camera (rotX, rotY in degrees).

    TRELLIS:
        eye = (r*cos(yaw)*cos(pitch), r*sin(yaw)*cos(pitch), r*sin(pitch))
        Z-up, perspective projection

    NMR orbit camera (from view matrix analysis):
        cam_pos = (-d*cos(A)*sin(B), -d*cos(A)*cos(B), d*sin(A))
        where A=rotX (rad), B=rotY (rad)

    Matching:  rotX = pitch,  rotY = -(yaw + π/2)
    """
    rotX = math.degrees(pitch)
    rotY = -math.degrees(yaw) - 90.0
    return rotX, rotY

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    xt_path = os.path.abspath(args.object)

    if not os.path.exists(xt_path):
        print(f"Error: not found: {xt_path}", file=sys.stderr)
        sys.exit(1)

    nmr_bin = _find_nmr()
    if nmr_bin is None:
        print("Error: NMR binary not found. Build with:\n"
              "  cd nmr && cmake -B build_qt -DCMAKE_BUILD_TYPE=Release "
              "-DPARASOLID_ROOT=... && cmake --build build_qt --target nmr",
              file=sys.stderr)
        sys.exit(1)

    with open(args.views) as f:
        views = json.load(f)
    if not views:
        print("Error: views.json is empty", file=sys.stderr)
        sys.exit(1)

    # Build NMR presets from TRELLIS views
    presets = []
    for i, v in enumerate(views):
        rotX, rotY = trellis_to_nmr_camera(v['yaw'], v['pitch'])
        presets.append({
            "name": f"{i:04d}",
            "rotX": rotX,
            "rotY": rotY,
        })

    # Write temporary presets file
    tmpdir = tempfile.mkdtemp(prefix='parasolid_render_')
    presets_path = os.path.join(tmpdir, 'presets.json')
    with open(presets_path, 'w') as f:
        json.dump({"presets": presets}, f)

    # Output format: {view}.png in tmpdir (NMR expands {view} to preset name)
    output_fmt = os.path.join(tmpdir, '{view}.png')

    # Run NMR snapshot
    env = dict(os.environ)
    dyld = _nmr_dyld()
    if dyld:
        env['DYLD_LIBRARY_PATH'] = dyld

    cmd = [
        nmr_bin, '--snapshot', xt_path,
        '--presets', presets_path,
        '--edges',
        '--no-burn-in',
        '-o', output_fmt,
        '--width', str(args.width),
        '--height', str(args.height),
    ]

    print(f"nmr:   {nmr_bin}", file=sys.stderr)
    print(f"input: {xt_path}", file=sys.stderr)
    print(f"views: {len(views)}", file=sys.stderr)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"NMR snapshot failed:\n{result.stderr}", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    # Move rendered images to expected output paths
    ok = 0
    for i, v in enumerate(views):
        src = os.path.join(tmpdir, f'{i:04d}.png')
        dst = v['output']
        if not os.path.isabs(dst):
            dst = os.path.abspath(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.move(src, dst)
            ok += 1
        else:
            print(f"  Warning: missing render for view {i}", file=sys.stderr)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Done: {ok}/{len(views)} views rendered.", file=sys.stderr)

if __name__ == '__main__':
    main()
