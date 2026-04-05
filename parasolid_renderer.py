#!/usr/bin/env python3
"""
Parasolid renderer — offscreen OpenGL renderer for Parasolid .x_t files.

Drop-in replacement for step_renderer. Loads the NMR-generated NPZ (vertices,
faces, edge_vertices) produced by `nmr-server --convert ... -o out.npz`, then
renders 512×512 RGBA images with Phong shading and B-Rep edge overlay.

CLI (same interface as step_renderer):
    python parasolid_renderer.py \\
        --object   path/to/part.x_t \\
        --views    path/to/views.json \\
        [--npz_dir /path/to/npz_files_dense] \\
        [--scale S] [--offset ox oy oz] \\
        [--output_dir /path/to/dataset/dir] \\
        [--width 512] [--height 512]

views.json format:
    [{"yaw": float, "pitch": float, "radius": float, "fov": float,
      "output": "path/to/NNN.png"}, ...]

NPZ lookup order:
    1. --npz_dir / {sha256}.npz
    2. abc_dataset/npz_files_dense/{sha256}.npz  (sibling of x_t_files/)
    3. Generate on the fly with nmr-server → /tmp/para_render_{sha[:16]}.npz
"""

import os, sys, json, argparse, hashlib, subprocess
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--object',     required=True, help='Path to .x_t file')
    p.add_argument('--views',      required=True, help='Path to views.json')
    p.add_argument('--npz_dir',    default=None,
                   help='Directory containing pre-built NMR NPZ files.')
    p.add_argument('--output_dir', default=None,
                   help='Dataset root (e.g. datasets/STEPFiles) — unused for rendering '
                        'but accepted for drop-in compatibility.')
    p.add_argument('--scale',      type=float, default=None,
                   help='Override normalization scale (applied after centering).')
    p.add_argument('--offset',     type=float, nargs=3, default=None,
                   help='Override center offset (x y z).')
    p.add_argument('--width',  type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    return p.parse_args()

# ---------------------------------------------------------------------------
# SHA256
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Mesh loading from NMR NPZ
# ---------------------------------------------------------------------------

def compute_vertex_normals(vertices, faces):
    """Area-weighted vertex normals via face normal accumulation."""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)            # [F,3], magnitude = 2*area
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return (normals / lens).astype(np.float32)

def normalize_mesh(vertices):
    """Normalize to [-0.5, 0.5]^3 (TRELLIS convention, same as prep_step_dataset.py)."""
    centroid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    v = vertices - centroid
    max_extent = np.abs(v).max()
    scale = (0.5 / max_extent * 0.95) if max_extent > 0 else 1.0
    return (v * scale).astype(np.float32)

def load_mesh_from_npz(npz_path, scale_override=None, offset_override=None):
    """
    Load an NMR-generated NPZ (vertices, faces, edge_vertices) and return
    geometry ready for upload to the GPU.

    Returns:
        vertices         [V,3] float32  (normalized to [-0.5, 0.5]^3)
        normals          [V,3] float32
        tri_indices      [F*3] uint32
        edge_seg_indices [E*2] uint32  (pairs of vertex indices, B-Rep edges)
    """
    data = np.load(npz_path)
    vertices = data['vertices'].astype(np.float64)  # world space
    faces    = data['faces'].astype(np.uint32)
    ev_set   = set(data['edge_vertices'].tolist())

    # Normalize (same transform as prep_step_dataset.py)
    if offset_override is not None:
        vertices -= np.array(offset_override, dtype=np.float64)
    else:
        centroid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        vertices -= centroid
    if scale_override is not None:
        vertices *= scale_override
    else:
        max_extent = np.abs(vertices).max()
        if max_extent > 0:
            vertices *= (0.5 / max_extent * 0.95)

    vertices = vertices.astype(np.float32)
    normals  = compute_vertex_normals(vertices, faces)
    tri_indices = faces.flatten()

    # Edge segments: mesh edges where both endpoints are B-Rep edge vertices
    seen, segs = set(), []
    for face in faces:
        for i in range(3):
            u, v = int(face[i]), int(face[(i + 1) % 3])
            key  = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                if u in ev_set and v in ev_set:
                    segs += [u, v]
    edge_seg_indices = np.array(segs, dtype=np.uint32) if segs else np.array([], dtype=np.uint32)

    return vertices, normals, tri_indices, edge_seg_indices

# ---------------------------------------------------------------------------
# Camera math (pure numpy)
# ---------------------------------------------------------------------------

def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r;  m[0, 3] = -np.dot(r, eye)
    m[1, :3] = u;  m[1, 3] = -np.dot(u, eye)
    m[2, :3] = -f; m[2, 3] =  np.dot(f, eye)
    return m

def perspective(fov_y, aspect, near, far):
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def camera_from_view(v):
    yaw, pitch = v['yaw'], v['pitch']
    radius, fov = v['radius'], v['fov']
    eye = np.array([
        radius * np.cos(yaw) * np.cos(pitch),
        radius * np.sin(yaw) * np.cos(pitch),
        radius * np.sin(pitch),
    ], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(eye / (np.linalg.norm(eye) + 1e-12), up)) > 0.999:
        up = np.array([0, 1, 0], dtype=np.float32)
    view = look_at(eye, np.zeros(3, dtype=np.float32), up)
    proj = perspective(fov, 1.0, 0.01, 100.0)
    return view, proj

# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

VERT_MESH = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 mvp;
uniform mat4 model_view;
uniform mat3 normal_mat;
out vec3 frag_pos_vs;
out vec3 frag_normal_vs;
void main() {
    vec4 mv_pos      = model_view * vec4(in_position, 1.0);
    frag_pos_vs      = mv_pos.xyz;
    frag_normal_vs   = normalize(normal_mat * in_normal);
    gl_Position      = mvp * vec4(in_position, 1.0);
}
"""

FRAG_MESH = """
#version 330 core
in vec3 frag_pos_vs;
in vec3 frag_normal_vs;
out vec4 out_color;

const vec3 L0   = normalize(vec3( 1.0,  1.0,  2.0));
const vec3 L1   = normalize(vec3(-1.5,  0.5,  1.0));
const vec3 L2   = normalize(vec3( 0.2, -1.0,  0.5));

const vec3  BASE = vec3(0.82, 0.83, 0.88);
const vec3  AMB  = vec3(0.18, 0.18, 0.20);
const vec3  SPEC = vec3(0.55, 0.55, 0.60);
const float SHIN = 52.0;

void main() {
    vec3 N = normalize(frag_normal_vs);
    vec3 V = normalize(-frag_pos_vs);
    if (!gl_FrontFacing) N = -N;

    vec3 c = AMB * BASE;
    // Key light
    float d0 = max(dot(N, L0), 0.0);
    vec3  R0 = reflect(-L0, N);
    float s0 = pow(max(dot(R0, V), 0.0), SHIN);
    c += d0 * 0.62 * BASE + s0 * 0.40 * SPEC;
    // Fill
    c += max(dot(N, L1), 0.0) * 0.28 * BASE;
    // Rim
    c += max(dot(N, L2), 0.0) * 0.12 * BASE;

    out_color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

# Edge quads: each segment is CPU-expanded to 4 vertices so we get proper
# thickness regardless of OpenGL max line width (= 1.0 on macOS).
#
# Per vertex attributes:
#   in_position  : 3D position of THIS endpoint
#   in_other     : 3D position of the OTHER endpoint
#   in_side      : -1.0 or +1.0 (which side of the ribbon)
VERT_EDGE_QUAD = """
#version 330 core
in vec3  in_position;
in vec3  in_other;
in float in_side;
uniform mat4  mvp;
uniform vec2  viewport;
uniform float half_width;   // pixels
void main() {
    vec4 a_clip = mvp * vec4(in_position, 1.0);
    vec4 b_clip = mvp * vec4(in_other,    1.0);

    vec2 a_ndc = a_clip.xy / a_clip.w;
    vec2 b_ndc = b_clip.xy / b_clip.w;

    // Direction in screen pixels (aspect-correct)
    vec2 dir = (b_ndc - a_ndc) * viewport * 0.5;
    float len = length(dir);
    if (len < 1e-6) { gl_Position = a_clip; return; }
    dir /= len;
    vec2 perp = vec2(-dir.y, dir.x);

    // Offset in clip space: half_width pixels -> NDC -> clip
    vec2 offset_ndc = perp * (2.0 * half_width / viewport);
    gl_Position = a_clip + vec4(offset_ndc * in_side * a_clip.w, 0.0, 0.0);

    // Depth bias: pull edges slightly toward camera to avoid z-fighting
    gl_Position.z -= 0.002 * gl_Position.w;
}
"""

FRAG_EDGE = """
#version 330 core
out vec4 out_color;
void main() {
    out_color = vec4(0.05, 0.06, 0.12, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, width=512, height=512, supersample=2, edge_px=1.5):
        import moderngl
        self.ctx = moderngl.create_standalone_context()
        self.w, self.h = width, height
        self.ss = supersample
        self.edge_px = edge_px

        sw, sh = width * supersample, height * supersample
        self.sw, self.sh = sw, sh

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((sw, sh), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((sw, sh))
        )
        self.prog_mesh = self.ctx.program(
            vertex_shader=VERT_MESH, fragment_shader=FRAG_MESH)
        self.prog_edge = self.ctx.program(
            vertex_shader=VERT_EDGE_QUAD, fragment_shader=FRAG_EDGE)

        self.has_edges = False
        self.vao_edge  = None

    def upload_mesh(self, vertices, normals, tri_indices, edge_seg_indices):
        # Interleaved position + normal
        vdata = np.hstack([vertices, normals]).astype(np.float32)
        self.vbo = self.ctx.buffer(vdata.tobytes())
        self.ibo = self.ctx.buffer(tri_indices.astype(np.uint32).tobytes())
        self.vao_mesh = self.ctx.vertex_array(
            self.prog_mesh,
            [(self.vbo, '3f 3f', 'in_position', 'in_normal')],
            self.ibo)

        self.has_edges = len(edge_seg_indices) >= 2
        if self.has_edges:
            segs  = edge_seg_indices.reshape(-1, 2)
            n     = len(segs)
            pos_a = vertices[segs[:, 0]]  # [N,3]
            pos_b = vertices[segs[:, 1]]  # [N,3]

            # 4 corners per segment: (a,-1),(a,+1),(b,+1),(b,-1)
            # Stack then interleave: shape [N,4,3] after transpose
            positions = np.stack([pos_a, pos_a, pos_b, pos_b], axis=1).reshape(-1, 3)
            others    = np.stack([pos_b, pos_b, pos_a, pos_a], axis=1).reshape(-1, 3)
            sides     = np.tile([-1.0, 1.0, 1.0, -1.0], n).astype(np.float32).reshape(-1, 1)
            edata     = np.hstack([positions, others, sides]).astype(np.float32)

            # Two triangles per quad: (0,1,2) and (0,2,3) in local coords
            base      = np.arange(n, dtype=np.uint32) * 4
            quad_tris = np.column_stack([
                base, base + 1, base + 2,
                base, base + 2, base + 3,
            ]).reshape(-1)

            self.ebo      = self.ctx.buffer(edata.tobytes())
            self.eibo     = self.ctx.buffer(quad_tris.tobytes())
            self.vao_edge = self.ctx.vertex_array(
                self.prog_edge,
                [(self.ebo, '3f 3f 1f', 'in_position', 'in_other', 'in_side')],
                self.eibo)

    def render_view(self, view_dict):
        import moderngl

        view_mat, proj_mat = camera_from_view(view_dict)
        mvp = (proj_mat @ view_mat).astype(np.float32)

        mv3 = view_mat[:3, :3]
        try:    normal_mat = np.linalg.inv(mv3).T.astype(np.float32)
        except: normal_mat = mv3.astype(np.float32)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.sw, self.sh)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        # Mesh pass
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        # OpenGL convention: column-major, so transpose our row-major matrices
        self.prog_mesh['mvp'].write(mvp.T.tobytes())
        self.prog_mesh['model_view'].write(view_mat.astype(np.float32).T.tobytes())
        self.prog_mesh['normal_mat'].write(normal_mat.tobytes())
        self.vao_mesh.render()

        # Edge pass
        if self.has_edges and self.vao_edge is not None:
            self.ctx.disable(moderngl.CULL_FACE)
            self.prog_edge['mvp'].write(mvp.T.tobytes())
            self.prog_edge['viewport'].value = (float(self.sw), float(self.sh))
            self.prog_edge['half_width'].value = self.edge_px * self.ss
            self.vao_edge.render(moderngl.TRIANGLES)

        raw = self.fbo.read(components=4)
        img = Image.frombytes('RGBA', (self.sw, self.sh), raw).transpose(Image.FLIP_TOP_BOTTOM)
        if self.ss > 1:
            img = img.resize((self.w, self.h), Image.LANCZOS)
        return img

    def close(self):
        self.ctx.release()

# ---------------------------------------------------------------------------
# Find pre-built or generate NMR NPZ
# ---------------------------------------------------------------------------

def _find_nmr():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'nmr', 'build', 'nmr-server'),
        '/Users/charlie/atomic/nmr/build/nmr-server',
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return 'nmr-server'

def _nmr_dyld():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'nmr', 'scripts', '_deps',
                        'parasolid_37_1_180', 'parasolid', 'arm_macos',
                        'base', 'shared_object')
    base = os.path.abspath(base)
    if os.path.exists(base):
        return base
    # Absolute fallback
    fb = ('/Users/charlie/atomic/nmr/scripts/_deps/'
          'parasolid_37_1_180/parasolid/arm_macos/base/shared_object')
    return fb if os.path.exists(fb) else ''

def find_npz(xt_path, sha, npz_dir=None):
    """Return path to NMR NPZ for this .x_t file, generating on the fly if needed."""
    # 1. Explicit npz_dir
    if npz_dir:
        p = os.path.join(npz_dir, f'{sha}.npz')
        if os.path.exists(p):
            return p

    # 2. Sibling npz_files_dense/ (abc_dataset layout)
    xt_abs   = os.path.abspath(xt_path)
    xt_dir   = os.path.dirname(xt_abs)
    xt_parent = os.path.dirname(xt_dir)
    dense    = os.path.join(xt_parent, 'npz_files_dense', f'{sha}.npz')
    if os.path.exists(dense):
        return dense

    # 3. Generate on the fly with nmr-server
    tmp = f'/tmp/para_render_{sha[:16]}.npz'
    if not os.path.exists(tmp):
        nmr  = _find_nmr()
        dyld = _nmr_dyld()
        env  = dict(os.environ, DYLD_LIBRARY_PATH=dyld)
        print(f"Generating NPZ via nmr-server ...", file=sys.stderr)
        result = subprocess.run(
            [nmr, '--convert', xt_abs, '--tess-max-width', '0.02', '-o', tmp],
            env=env, capture_output=True)
        if result.returncode != 0:
            print(result.stderr.decode(errors='replace'), file=sys.stderr)
    return tmp if os.path.exists(tmp) else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.object):
        print(f"Error: not found: {args.object}", file=sys.stderr)
        sys.exit(1)

    with open(args.views) as f:
        views = json.load(f)
    if not views:
        print("Error: views.json is empty", file=sys.stderr)
        sys.exit(1)

    sha = sha256_file(args.object)
    npz_path = find_npz(args.object, sha, npz_dir=args.npz_dir)

    if npz_path is None:
        print(f"Error: could not find or generate NPZ for {args.object}", file=sys.stderr)
        sys.exit(1)

    print(f"npz:   {npz_path}", file=sys.stderr)

    vertices, normals, tri_indices, edge_seg_indices = load_mesh_from_npz(
        npz_path,
        scale_override=args.scale,
        offset_override=args.offset,
    )
    n_faces = len(tri_indices) // 3
    n_edges = len(edge_seg_indices) // 2
    print(f"  {len(vertices)} verts  {n_faces} faces  {n_edges} B-Rep edge segs",
          file=sys.stderr)

    renderer = Renderer(args.width, args.height, supersample=2, edge_px=1.5)
    renderer.upload_mesh(vertices, normals, tri_indices, edge_seg_indices)

    for i, view in enumerate(views):
        out = view['output']
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        img = renderer.render_view(view)
        img.save(out)
        if i == 0 or (i + 1) % 20 == 0 or i == len(views) - 1:
            print(f"  [{i+1}/{len(views)}] {os.path.basename(out)}", file=sys.stderr)

    renderer.close()
    print(f"Done: {len(views)} views.", file=sys.stderr)

if __name__ == '__main__':
    main()
