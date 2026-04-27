FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/workspace/.cache/huggingface

# ---- System packages (stable) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build ca-certificates curl \
      libgl1 libglib2.0-0 libjpeg-dev ffmpeg \
      python3.10 python3.10-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# ---- uv + venv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /workspace/TRELLIS
RUN uv venv /opt/venv --python 3.10
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ---- PyTorch (stable) ----
RUN uv pip install torch==2.4.0 torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cu118

# ---- Prebuilt CUDA wheels (fast; batch together) ----
RUN uv pip install xformers==0.0.27.post2 \
      --index-url https://download.pytorch.org/whl/cu118 \
 && uv pip install psutil packaging \
 && uv pip install flash-attn --no-build-isolation \
 && uv pip install spconv-cu118 \
 && uv pip install kaolin \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# ---- Source-built CUDA kernels (slow; isolate each so app edits don't rebuild) ----
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
 && uv pip install /tmp/nvdiffrast --no-build-isolation \
 && rm -rf /tmp/nvdiffrast

RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast \
 && uv pip install /tmp/diffoctreerast --no-build-isolation \
 && rm -rf /tmp/diffoctreerast

RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting \
 && uv pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation \
 && rm -rf /tmp/mip-splatting

# ---- Python deps (basic + train) ----
RUN uv pip install \
      pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
      rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers \
      tensorboard pandas lpips \
 && uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# ---- App source (changes often; keep last) ----
COPY . /workspace/TRELLIS

# Fail fast if the FlexiCubes submodule wasn't initialized on the host.
# trellis/representations/mesh/cube2mesh.py:5 imports from .flexicubes.flexicubes
# which only exists if `git submodule update --init --recursive` has been run.
RUN test -f trellis/representations/mesh/flexicubes/flexicubes.py \
 || (echo "ERROR: FlexiCubes submodule missing. Run 'git submodule update --init --recursive' on the host before building." && exit 1)

CMD ["bash"]
