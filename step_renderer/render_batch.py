import subprocess
import numpy as np
import json
import os
import sys
import glob

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

def render_step(step_file, output_dir, num_views=150, renderer=None):
    if renderer is None:
        renderer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "step_render")
    os.makedirs(output_dir, exist_ok=True)
    offset = (np.random.rand(), np.random.rand())
    fov = 40 / 180 * np.pi

    views = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        views.append({
            "yaw": yaw, "pitch": pitch,
            "radius": 2.0, "fov": fov,
            "output": os.path.join(output_dir, f"view_{i:04d}.png")
        })

    views_file = os.path.join(output_dir, "views.json")
    with open(views_file, "w") as f:
        json.dump(views, f)

    subprocess.run([
        renderer,
        "--object", step_file,
        "--views", views_file
    ], check=True)

def batch_render(step_dir, output_root, num_views=150, renderer=None):
    if renderer is None:
        renderer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "step_render")
    step_files = sorted(glob.glob(os.path.join(step_dir, "*.step")) +
                        glob.glob(os.path.join(step_dir, "*.stp")))
    print(f"Found {len(step_files)} STEP files")

    for i, step_file in enumerate(step_files):
        name = os.path.splitext(os.path.basename(step_file))[0]
        output_dir = os.path.join(output_root, name)
        print(f"\n[{i+1}/{len(step_files)}] {name}")
        try:
            render_step(step_file, output_dir, num_views, renderer)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_batch.py <step_dir> <output_dir> [num_views]")
        sys.exit(1)
    nv = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    batch_render(sys.argv[1], sys.argv[2], nv)