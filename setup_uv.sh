#!/bin/bash
# UV-based setup script for TRELLIS (mirrors setup.sh)

# Read Arguments
TEMP=$(getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo -n 'setup_uv.sh' -- "$@")

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
SPCONV=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false
ERROR=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: source setup_uv.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new uv virtual environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    echo ""
    echo "Example (full install for training with CUDA 11.8):"
    echo "  export PATH=/usr/local/cuda-11.8/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH"
    echo "  export CUDA_HOME=/usr/local/cuda-11.8"
    echo "  source setup_uv.sh --new-env --basic --train --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast"
    return 0 2>/dev/null || exit 0
fi

# Check if uv is installed
if ! command -v uv &> /dev/null ; then
    echo "[ERROR] uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1 2>/dev/null || exit 1
fi

WORKDIR=$(pwd)

if [ "$NEW_ENV" = true ] ; then
    echo "[UV] Creating new virtual environment with Python 3.10..."
    uv venv .venv --python 3.10
    source .venv/bin/activate
    
    echo "[UV] Installing PyTorch 2.4.0 with CUDA 11.8..."
    uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Activate venv if it exists and not already activated
if [ -z "$VIRTUAL_ENV" ] && [ -d ".venv" ] ; then
    echo "[UV] Activating existing virtual environment..."
    source .venv/bin/activate
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] ; then
    echo "[ERROR] No virtual environment active. Run with --new-env or activate .venv first."
    return 1 2>/dev/null || exit 1
fi

# Get system information
PYTORCH_VERSION_FULL=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -z "$PYTORCH_VERSION_FULL" ] ; then
    echo "[ERROR] PyTorch not installed. Run with --new-env first."
    return 1 2>/dev/null || exit 1
fi
# Strip the +cuXXX suffix for version matching (e.g., 2.4.0+cu118 -> 2.4.0)
PYTORCH_VERSION=$(echo $PYTORCH_VERSION_FULL | cut -d'+' -f1)

PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    *)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, Platform: $PLATFORM"
        ;;
esac

if [ "$BASIC" = true ] ; then
    echo "[UV] Installing basic dependencies..."
    uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
    uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

if [ "$TRAIN" = true ] ; then
    echo "[UV] Installing training dependencies..."
    uv pip install tensorboard pandas lpips
    # Note: pillow-simd requires libjpeg-dev: sudo apt install -y libjpeg-dev
    # Uncomment below if you want pillow-simd (faster but requires system deps)
    # uv pip uninstall pillow
    # uv pip install pillow-simd
fi

if [ "$XFORMERS" = true ] ; then
    echo "[UV] Installing xformers..."
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ "$CUDA_VERSION" = "11.8" ] ; then
            case $PYTORCH_VERSION in
                2.0.1) uv pip install https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl ;;
                2.1.0) uv pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.1) uv pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.2) uv pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.0) uv pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.1) uv pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.2) uv pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.3.0) uv pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.0) uv pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.1) uv pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.5.0) uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ] ; then
            case $PYTORCH_VERSION in
                2.1.0) uv pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.1) uv pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.2) uv pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.0) uv pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.1) uv pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.2) uv pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.3.0) uv pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.0) uv pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.1) uv pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.5.0) uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ] ; then
            case $PYTORCH_VERSION in
                2.5.0) uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_VERSION"
        fi
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$FLASHATTN" = true ] ; then
    echo "[UV] Installing flash-attn..."
    if [ "$PLATFORM" = "cuda" ] ; then
        # flash-attn build requires psutil
        uv pip install psutil packaging
        uv pip install flash-attn --no-build-isolation
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$KAOLIN" = true ] ; then
    echo "[UV] Installing kaolin..."
    if [ "$PLATFORM" = "cuda" ] ; then
        case $PYTORCH_VERSION in
            2.0.1) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html ;;
            2.1.0) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html ;;
            2.1.1) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html ;;
            2.2.0) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html ;;
            2.2.1) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html ;;
            2.2.2) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html ;;
            2.4.0) uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html ;;
            *) echo "[KAOLIN] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    echo "[UV] Installing nvdiffrast..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        if [ ! -d "/tmp/extensions/nvdiffrast" ] ; then
            git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        fi
        uv pip install /tmp/extensions/nvdiffrast --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DIFFOCTREERAST" = true ] ; then
    echo "[UV] Installing diffoctreerast..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        if [ ! -d "/tmp/extensions/diffoctreerast" ] || [ ! -f "/tmp/extensions/diffoctreerast/lib/glm/glm/glm.hpp" ] ; then
            rm -rf /tmp/extensions/diffoctreerast
            git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
        fi
        uv pip install /tmp/extensions/diffoctreerast --no-build-isolation
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$MIPGAUSSIAN" = true ] ; then
    echo "[UV] Installing mip-splatting..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        if [ ! -d "/tmp/extensions/mip-splatting" ] ; then
            git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
        fi
        uv pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$VOX2SEQ" = true ] ; then
    echo "[UV] Installing vox2seq..."
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ -d "extensions/vox2seq" ] ; then
            mkdir -p /tmp/extensions
            cp -r extensions/vox2seq /tmp/extensions/vox2seq
            uv pip install /tmp/extensions/vox2seq
        else
            echo "[VOX2SEQ] extensions/vox2seq not found, skipping..."
        fi
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$SPCONV" = true ] ; then
    echo "[UV] Installing spconv..."
    if [ "$PLATFORM" = "cuda" ] ; then
        case $CUDA_MAJOR_VERSION in
            11) uv pip install spconv-cu118 ;;
            12) uv pip install spconv-cu120 ;;
            *) echo "[SPCONV] Unsupported CUDA major version: $CUDA_MAJOR_VERSION" ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DEMO" = true ] ; then
    echo "[UV] Installing demo dependencies..."
    uv pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
fi

echo ""
echo "[UV] Setup complete!"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "PyTorch: $PYTORCH_VERSION"
if [ "$PLATFORM" = "cuda" ] ; then
    echo "CUDA: $CUDA_VERSION"
fi
