SHELL := /bin/bash

# --- Configuration (override on CLI: SUBSET=Foo make voxelize) ---
NMR_DIR      ?= ../nmr
DATASETS_DIR ?= $(abspath ./datasets)
OUTPUTS_DIR  ?= $(abspath ./outputs)
SUBSET       ?= ParasolidDemo
DATASET      ?= ParasolidFiles
CUDA_VISIBLE_DEVICES ?=

# Files in dataset_toolkits/datasets/ register the dataset (ParasolidFiles.py,
# ABO.py, ObjaverseXL.py, etc.). voxelize.py and build_metadata.py take this
# module name as sys.argv[1]. extract_feature/encode_latent/encode_ss_latent
# do not.

export DATASETS_DIR OUTPUTS_DIR

COMPOSE_RUN := docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
  trellis

.PHONY: help
help:
	@echo "Build:"
	@echo "  build              build TRELLIS image"
	@echo "  build-nmr          build NMR image (delegates to $(NMR_DIR))"
	@echo "  build-all          both"
	@echo "Shells:"
	@echo "  shell              bash inside TRELLIS"
	@echo "  shell-nmr          bash inside NMR"
	@echo "Data prep (Parasolid):"
	@echo "  nmr-render         SUBSET=$(SUBSET)  render .x_t multi-views (NMR)"
	@echo "  nmr-tessellate     SUBSET=$(SUBSET)  mesh + UDF (NMR)"
	@echo "  voxelize           SUBSET=$(SUBSET)  256^3 voxels (TRELLIS)"
	@echo "  extract-features   SUBSET=$(SUBSET)  DINOv2 features (TRELLIS, GPU)"
	@echo "  encode-latents     SUBSET=$(SUBSET)  SLAT latents (TRELLIS, GPU)"
	@echo "  encode-ss-latents  SUBSET=$(SUBSET)  SS latents (TRELLIS, GPU)"
	@echo "  build-metadata     SUBSET=$(SUBSET)"
	@echo "  data-prep          all TRELLIS-side prep, in order"
	@echo "Train:"
	@echo "  train CONFIG=path/to.json [SUBSET=...]"
	@echo "Maintenance:"
	@echo "  smoke              GPU + import sanity check"
	@echo "  clean              prune dangling images (asks first)"

# --- Build ---
.PHONY: build build-nmr build-all submodules
submodules:
	git submodule update --init --recursive
build: submodules
	docker compose build trellis
build-nmr:
	$(MAKE) -C $(NMR_DIR) build
build-all: build build-nmr

# --- Shells ---
.PHONY: shell shell-nmr
shell:
	$(COMPOSE_RUN) bash
shell-nmr:
	$(MAKE) -C $(NMR_DIR) shell

# --- NMR-side prep (delegates) ---
.PHONY: nmr-render nmr-tessellate
nmr-render:
	$(MAKE) -C $(NMR_DIR) render SUBSET=$(SUBSET) DATASETS_DIR=$(DATASETS_DIR)
nmr-tessellate:
	$(MAKE) -C $(NMR_DIR) tessellate SUBSET=$(SUBSET) DATASETS_DIR=$(DATASETS_DIR)

# --- TRELLIS-side prep ---
# Note: voxelize.py and build_metadata.py take DATASET as sys.argv[1] (positional
# before flag args). extract_feature/encode_latent/encode_ss_latent do not.
.PHONY: voxelize extract-features encode-latents encode-ss-latents build-metadata data-prep
voxelize:
	$(COMPOSE_RUN) python dataset_toolkits/voxelize.py $(DATASET) \
	  --output_dir /workspace/datasets/$(SUBSET)
extract-features:
	$(COMPOSE_RUN) python dataset_toolkits/extract_feature.py \
	  --output_dir /workspace/datasets/$(SUBSET)
encode-latents:
	$(COMPOSE_RUN) python dataset_toolkits/encode_latent.py \
	  --output_dir /workspace/datasets/$(SUBSET)
encode-ss-latents:
	$(COMPOSE_RUN) python dataset_toolkits/encode_ss_latent.py \
	  --output_dir /workspace/datasets/$(SUBSET)
build-metadata:
	$(COMPOSE_RUN) python dataset_toolkits/build_metadata.py $(DATASET) \
	  --output_dir /workspace/datasets/$(SUBSET)

data-prep: voxelize extract-features encode-latents encode-ss-latents build-metadata

# --- Train ---
.PHONY: train
train:
ifndef CONFIG
	$(error CONFIG=path/to/config.json is required)
endif
	$(COMPOSE_RUN) python train.py \
	  --config $(CONFIG) \
	  --data_dir /workspace/datasets/$(SUBSET) \
	  --output_dir /workspace/outputs

# --- Smoke / health ---
.PHONY: smoke
smoke:
	$(COMPOSE_RUN) bash -lc 'nvidia-smi && python -c "import torch, nvdiffrast.torch, diff_gaussian_rasterization, flash_attn, spconv, kaolin, trellis; print(\"ok\", torch.cuda.is_available(), torch.version.cuda)"'

# --- Cleanup ---
.PHONY: clean
clean:
	@read -p "Prune dangling images and stopped containers? [y/N] " ans; [[ $$ans == y ]] || exit 1
	docker system prune -f
