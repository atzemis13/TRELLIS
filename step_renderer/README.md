# STEP File Batch Renderer

Offscreen renderer for STEP CAD files using OpenCASCADE (OCCT). Produces 1024×1024 PNG images with transparent backgrounds. Renders shaded geometry with black face boundary edges and blue B-spline knot lines.

Originally written for macOS (Cocoa), ported to Linux (X11) for use in WSL2.

## Features

- Loads a STEP file, normalizes it to a unit bounding box centered at the origin
- Renders shaded faces with black boundary edges
- Extracts and renders B-spline knot isocurves in blue (boundary knots filtered out)
- Magenta chroma key background converted to transparency in output PNG
- Supports single view (CLI args) or batch views (JSON file)
- Hammersley sphere sampling for uniform camera coverage

## Prerequisites

Ubuntu 22.04 (WSL2). Needs `DISPLAY=:0` set (WSLg provides this).

### Install dependencies

```bash
sudo apt install -y \
  build-essential cmake \
  libocct-data-exchange-dev \
  libocct-draw-dev \
  libocct-foundation-dev \
  libocct-modeling-algorithms-dev \
  libocct-modeling-data-dev \
  libocct-visualization-dev \
  libx11-dev libxext-dev libxmu-dev \
  libgl1-mesa-dev libglu1-mesa-dev \
  libfreetype-dev libfreeimage-dev \
  xvfb
```

### Get stb_image_write.h

```bash
cd ~/atomic/TRELLIS/Archive
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

## Build

```bash
cd ~/atomic/TRELLIS/Archive
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### CMakeLists.txt

The Ubuntu 22.04 OCCT packages don't ship a top-level `OpenCASCADEConfig.cmake`, so `find_package` won't work. The CMakeLists hardcodes include/lib paths instead:

```cmake
cmake_minimum_required(VERSION 3.16)
project(step_render CXX)

add_executable(step_render render.cpp)

target_include_directories(step_render PRIVATE
  /usr/include/opencascade
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(step_render PRIVATE
  TKernel TKMath TKBRep TKGeomBase TKGeomAlgo
  TKTopAlgo TKShHealing TKG2d TKG3d
  TKV3d TKOpenGl TKService
  TKSTEP TKSTEPBase TKSTEPAttr TKSTEP209 TKXSBase
  TKMesh TKHLR TKPrim
  X11 Xext GL
)
```

## Usage

### Single view

```bash
export DISPLAY=:0
./build/step_render \
  --object /path/to/model.step \
  --output output.png \
  --yaw 0.5 --pitch 0.3 --radius 2 --fov 0.698
```

- `--yaw` / `--pitch`: camera rotation in radians
- `--radius`: camera distance (model is normalized to unit size)
- `--fov`: field of view in radians (0.698 ≈ 40°)

### Batch views (JSON)

```bash
./build/step_render \
  --object /path/to/model.step \
  --views views.json
```

Where `views.json` looks like:

```json
[
  {"yaw": 0.0, "pitch": 0.1, "radius": 2.0, "fov": 0.698, "output": "out/view_0000.png"},
  {"yaw": 1.2, "pitch": -0.3, "radius": 2.0, "fov": 0.698, "output": "out/view_0001.png"}
]
```

### Batch render many STEP files (Python)

```bash
python3 render_batch.py <step_dir> <output_dir> [num_views]
```

Example:

```bash
python3 render_batch.py ./step_files ./renders 150
```

This will:
1. Find all `.step` / `.stp` files in `step_dir`
2. For each file, generate 150 camera views using Hammersley sphere sampling
3. Write a `views.json` and call `step_render` once per file
4. Output PNGs to `<output_dir>/<filename>/view_NNNN.png`

## Camera Sampling

Views are generated using Hammersley quasi-random sampling on the sphere, giving uniform coverage. A random offset is applied per STEP file so each model gets a slightly different set of viewpoints. The sampling produces:

- `phi` (yaw): `[0, 2π]`
- `theta` (pitch): `[-π/2, π/2]`
- Fixed radius of 2 and FOV of 40°

## Notes

- Output images are 1024×1024 RGBA PNGs
- Background transparency is achieved via magenta chroma keying (pixels matching the magenta background within a tolerance are set to alpha 0)
- The OCCT default red isoparametric lines are disabled; only the explicitly drawn knot lines and face boundaries are shown
- If `DISPLAY` is not set, WSLg may not be running — install `xvfb` and run `Xvfb :99 &` then `export DISPLAY=:99` as a fallback