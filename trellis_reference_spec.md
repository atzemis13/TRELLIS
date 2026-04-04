Note: This was transcribed and  from the original Trellis paper for LLM consumption. 

# Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS)

## 1. Core Concept

TRELLIS uses a unified **Structured LATent (SLAT)** representation that encodes both geometry and appearance of 3D assets into localized latents on a sparse 3D grid. Different decoders can map SLAT to diverse output formats: 3D Gaussians, Radiance Fields, and meshes.

Key properties:
- No 3D fitting needed for training objects (uses multiview rendering + DINOv2 features instead)
- Sparse structure means L ≪ N³ active voxels, enabling high-resolution grids efficiently
- Locality of latents enables flexible editing (region-specific operations)
- The encoder is trained end-to-end with the Gaussian decoder; other decoders are trained by freezing the encoder

---

## 2. Structured Latent Representation (SLAT)

For a 3D asset O, the representation is:

```
z = {(zᵢ, pᵢ)}  for i = 1..L

where:
  pᵢ ∈ {0, 1, ..., N-1}³   — positional index of an active voxel in the 3D grid
  zᵢ ∈ ℝᶜ                  — local latent vector attached to that voxel
  N = 64                    — spatial length of the 3D grid (default)
  L ≈ 20K                   — average number of active voxels (L ≪ N³ = 262144)
  C = 8                     — latent channel dimension (from Table 3 ablation: 64³ × 8 is default)
```

**Active voxels:** Paper states pᵢ is "the positional index of an active voxel in the 3D grid intersecting with the surface of O". The exact voxelization method (from mesh, from depth maps, etc.) is not described — check code.

**Latents** `zᵢ` capture finer details: "the active voxels pᵢ outline the coarse structure of the 3D asset, while the latents zᵢ capture finer details of appearance and shape."

---

## 3. Encoding: 3D Asset → SLAT

### 3.1 Visual Feature Aggregation

1. Render 150 images of the 3D asset from randomly sampled camera views on a sphere (radius=2, FoV=40°, smooth area lighting via Blender)
2. Extract feature maps from each view using a **pretrained DINOv2 encoder** (specific variant not stated in paper — check code)
3. For each active voxel, project it onto the multiview feature maps, retrieve features at corresponding locations
4. "Their average is used as fᵢ"

This produces a voxelized feature volume:
```
f = {(fᵢ, pᵢ)}  for i = 1..L
```
at the same resolution as SLAT (64³).

**Note:** The dimension of fᵢ (the DINOv2 feature vector per voxel) is not explicitly stated — check code for the encoder input projection.

### 3.2 Sparse VAE Encoder

The encoder `E` maps the voxelized features `f` to structured latents `z`:

```
E: f → z
```

**Architecture (Table 6):**
- 12 transformer layers, dim=768, 12 heads
- Block architecture: `3D-SW-MSA + FFN` (3D Shifted Window Multi-head Self-Attention)
- Parameters: 85.8M
- Special module: 3D Swin Attention

**How it processes sparse data:**
1. Serialize input features from active voxels into a token sequence
2. Add sinusoidal positional encodings based on voxel positions `pᵢ`
3. This creates a variable-length sequence (length = L ≈ 20K)
4. Process through transformer blocks with shifted window attention

**3D Shifted Window Attention details:**
- "We partition the 64³ space into 8³ windows" — likely means window size = 8 in each dimension (consistent with shift of (4,4,4) being half the window size), giving (64/8)³ = 512 windows. Verify in code.
- Tokens inside each window perform self-attention independently
- "Despite the potential variation in the number of tokens per window, this challenge can be efficiently addressed using modern attention implementations (e.g., FlashAttention and xformers)"
- "The transformer blocks alternate between non-shifted window attention and window attention shifted by (4, 4, 4), ensuring that the windows in adjacent layers overlap uniformly"

**KL regularization** is applied to zᵢ to encourage normal distribution (standard VAE formulation following Latent Diffusion Models).

---

## 4. Decoding: SLAT → 3D Representations

All decoders share the same transformer backbone architecture as the encoder (12 layers, dim=768, 12 heads, 3D Swin Attention) but differ in output layers and losses.

### 4.1 Decoder: 3D Gaussians (D_GS)

**This is the primary decoder — the encoder is trained end-to-end with this one.**

```
D_GS: {(zᵢ, pᵢ)} → {{(oᵏᵢ, cᵏᵢ, sᵏᵢ, αᵏᵢ, rᵏᵢ)} for k=1..K} for i=1..L
```

Each zᵢ is decoded into K=32 Gaussians with:
- `oᵏᵢ` — position offsets
- `cᵏᵢ` — colors
- `sᵏᵢ` — scales
- `αᵏᵢ` — opacities
- `rᵏᵢ` — rotations

**Locality constraint:** Final positions are `xᵏᵢ = pᵢ + tanh(oᵏᵢ)` (constrains Gaussians to vicinity of their voxel).

**Parameters:** 85.4M

**Loss function:**
```
L_GS = L_recon + L_vol + L_α

where:
  L_recon = L1 + 0.2*(1 - SSIM) + 0.2*LPIPS
  L_vol   = (1/LK) * Σᵢ Σₖ Π(sᵏᵢ)          — volume regularization (prevents excessively large Gaussians)
  L_α     = (1/LK) * Σᵢ Σₖ (1 - αᵏᵢ)²       — opacity regularization (prevents transparent Gaussians)
```

**Additional settings (from Mip-Splatting):**
- Minimal Gaussian scale: 9e-4 (derived from 512³ sampling rate in (-0.5, 0.5)³ cube)
- Screen space Gaussian filter variance: 0.1

### 4.2 Decoder: Radiance Fields (D_RF)

```
D_RF: {(zᵢ, pᵢ)} → {(vˣᵢ, vʸᵢ, vᶻᵢ, vᶜᵢ)} for i=1..L
```

Each zᵢ is decoded into 4 orthogonal vectors representing a **CP-decomposition** of a local 8³ radiance volume:

```
vˣᵢ, vʸᵢ, vᶻᵢ ∈ ℝ^(16×8)
vᶜᵢ ∈ ℝ^(16×4)

Rank R = 16

Local volume V ∈ ℝ^(8×8×8×4):
  V[x,y,z,c] = Σᵣ vˣᵢ[r,x] * vʸᵢ[r,y] * vᶻᵢ[r,z] * vᶜᵢ[r,c]
```

The last dimension (size 4) contains color and density. Local volumes are assembled by voxel positions into a 512³ radiance field.

**Parameters:** 85.4M
**Loss:** L_recon (same as Gaussians: L1 + 0.2*(1-SSIM) + 0.2*LPIPS)

Includes a custom CUDA differentiable renderer that integrates sorting, ray marching, radiance integration, and CP reconstruction in one kernel.

### 4.3 Decoder: Meshes (D_M)

```
D_M: {(zᵢ, pᵢ)} → {{(wʲᵢ, dʲᵢ, cʲᵢ, nʲᵢ)} for j=1..64} for i=1..L
```

**Key difference from other decoders:** Two sparse convolutional upsampling blocks are appended after the transformer backbone to increase output resolution from 64³ to 256³ (each zᵢ covers a 4³ sub-grid).

**Per high-resolution active voxel outputs:**
- `wʲᵢ` — FlexiCubes flexible parameters, consisting of:
  - `αʲᵢ ∈ ℝ⁸` — interpolation weights per voxel
  - `βʲᵢ ∈ ℝ¹²` — interpolation weights per voxel
  - `γʲᵢ ∈ ℝ¹` — splitting weights per voxel
  - `δʲᵢ ∈ ℝ^(8×3)` — per vertex deformation vectors
- `dʲᵢ ∈ ℝ⁸` — signed distance values for 8 vertices of the voxel
- `cʲᵢ ∈ ℝ^(8×3)` — vertex colors
- `nʲᵢ ∈ ℝ^(8×3)` — vertex normals

Since each vertex connects to multiple voxels, final vertex attributes (δ, d, c, n) are averaged across all associated voxels.

**Mesh extraction:** Paper states: "we attach the sparse structure to a dense grid for differentiable surface extraction using FlexiCubes. For all inactive voxels in the dense grid, we set their signed distance values to 1.0 and all other associated attributes to zero. We then extract meshes from the 0-level iso-surfaces of the dense grid." Vertex attributes (c, n) are interpolated from grid vertices. Rendering uses **Nvdiffrast**.

**Note:** The paper does not describe how vertex attributes are interpolated from the grid — likely FlexiCubes' built-in interpolation. Check code.

**Parameters:** 90.9M (extra from Sparse Conv Upsampler)

**Rendered outputs:** foreground mask M, depth map D, mesh-derived normal map Nm, RGB image C, predicted normal map N.

**Loss function:**
```
L_M = L_geo + 0.1*L_color + L_reg

where:
  L_geo   = L1(M) + 10*L_Huber(D) + L_recon(Nm)
  L_color = L_recon(C) + L_recon(N)
  L_recon = L1 + 0.2*(1-SSIM) + 0.2*LPIPS   (same as before)

  L_reg = L_consist + L_dev + 0.01*L_tsdf
    L_consist — penalizes variance of attributes for the same voxel vertex
    L_dev     — FlexiCubes regularization for plausible mesh extraction
    L_tsdf    — enforces SDF values to match distances between grid vertices and extracted surface (stabilizes early training)
```

---

## 5. Training Protocol for Decoders

**Critical workflow:**
1. Train encoder `E` + Gaussian decoder `D_GS` end-to-end (primary training)
2. **Freeze the encoder** `E`
3. Train other decoders (`D_RF`, `D_M`) from scratch using the frozen encoder's latents

This means the structured latents are defined by the Gaussian reconstruction objective, and other decoders must learn to decode from those same latents. This works well empirically (Table 1 shows strong reconstruction across all formats).

---

## 6. Sparse Structure VAE (for Generation Pipeline)

Separate from the SLAT VAE above, there's a small VAE for compressing the binary occupancy grid for the generation stage:

**Encoder E_S / Decoder D_S:**
- 3D Convolutional U-Net architecture (similar to LDM VAEs but with 3D convolutions, no self-attention)
- Compresses binary grid O ∈ {0,1}^(N×N×N) → feature grid S ∈ ℝ^(D×D×D×C_S)
- Spatial compression: 64³ → 16³
- Feature channels: 32 (at 64³), 128 (at 32³), 512 (at 16³)
- Latent channel dimension: 8
- Uses pixel shuffle in upsampling, layer normalization instead of group normalization
- Trained with **Dice loss** (handles imbalance between active/inactive voxels)
- Parameters: E_S = 59.3M, D_S = 73.7M
- Nearly lossless compression since O represents only coarse geometry

---

## 7. Generation Pipeline (Two-Stage)

Uses rectified flow transformers. Forward process: `x(t) = (1-t)*x₀ + t*ε`. CFM objective: `L = E[||vθ(x,t) - (ε - x₀)||²]`.

### Stage 1: Sparse Structure Generation (G_S)

Generates the feature grid S (decoded to binary occupancy grid O, then to active voxels {pᵢ}).

**Note:** The paper does not describe how the continuous output of D_S is thresholded back to the discrete binary grid O — check code for the binarization method.

**Architecture (Table 6):** Standard transformer (not sparse).
- Serialized dense noisy grid + positional encodings
- adaLN + gating for timestep conditioning
- Cross-attention for text (**CLIP** features — specific variant not stated) or image (**DINOv2** features) conditions
- Sizes: B=157M, L=543M, XL=975M (text); L=556M (image)

### Stage 2: Structured Latents Generation (G_L)

Generates latents {zᵢ} given structure {pᵢ}.

**Architecture (Table 6):** Sparse flow transformer.
- Sparse convolution downsampler packs latents from 2³ local regions (64³→32³)
- Time-modulated transformer blocks
- Sparse convolution upsampler at end + skip connections to downsampler
- adaLN for timesteps, cross-attention for conditions
- **Sizes:** B=185M, L=588M, XL=1073M (text); L=600M (image)

**Sparse Conv Downsampler/Upsampler details (for G_L):**
- "These blocks are composed of residual networks with two sparse convolutional layers, skip connections with optional linear mappings, and pooling or unpooling operators"
- Downsampling: average pooling — "we only average the features from active voxels within each 2³ pooling window"
- Upsampling: nearest-neighbor unpooling — "assigning values to active voxels from their nearest neighbors in the 32³ space"
- For D_M upsampling: "we simply subdivide each voxel into 2³, resulting in a new sparse tensor with doubled spatial dimensions in each upsampling block"
- "Given that the structures of 64³ are pre-determined, we only average the features from active voxels within each 2³ pooling window and recover the 64³ structures during unpooling"

### Training settings:
- Classifier-free guidance (CFG) drop rate: 0.1
- Optimizer: AdamW, lr=1e-4
- Timestep sampling: logitNorm(1, 1) — better than logitNorm(0, 1) for this task
- XL model: 64 A100 GPUs (40G), 400K steps, batch size 256

### Inference settings:
- CFG strength: 3
- Sampling steps: 50

---

## 8. Network Configuration Summary (Table 6)

| Network | #Layer | #Dim | #Head | Block | Special Modules | #Param |
|---------|--------|------|-------|-------|-----------------|--------|
| E_S (struct VAE enc) | — | — | — | — | 3D Conv U-Net | 59.3M |
| D_S (struct VAE dec) | — | — | — | — | 3D Conv U-Net | 73.7M |
| E (SLAT encoder) | 12 | 768 | 12 | 3D-SW-MSA+FFN | 3D Swin Attn | 85.8M |
| D_GS (Gaussian dec) | 12 | 768 | 12 | 3D-SW-MSA+FFN | 3D Swin Attn | 85.4M |
| D_RF (RadField dec) | 12 | 768 | 12 | 3D-SW-MSA+FFN | 3D Swin Attn | 85.4M |
| D_M (Mesh dec) | 12 | 768 | 12 | 3D-SW-MSA+FFN | 3D Swin Attn + Sp.Conv Upsampler | 90.9M |
| G_S (struct gen) | 12-28 | 768-1280 | 12-16 | MSA+MCA+FFN | QK Norm | 157M-975M |
| G_L (latent gen) | 12-28 | 768-1280 | 12-16 | MSA+MCA+FFN | QK Norm + Sp.Conv Down/Up + Skip | 185M-1073M |

---

## 9. Data

- ~500K high-quality 3D assets from Objaverse-XL, ABO, 3D-FUTURE, HSSD
- Filtered by aesthetic score threshold (5.5 for Objaverse-XL, 4.5 for others)
- 150 rendered images per asset for VAE training
- GPT-4o captioning for text prompts
- Image prompts: rendered with augmented FoVs (10°-70°)
- Evaluation: Toys4k dataset (not in training set)

---

## 10. Key Ablation Results

**SLAT size (Table 3):** 64³ with 8 channels significantly outperforms 32³ with up to 64 channels.

| Resolution | Channel | PSNR↑ | LPIPS↓ |
|-----------|---------|-------|--------|
| 32 | 16 | 31.64 | 0.0297 |
| 32 | 32 | 31.80 | 0.0289 |
| 32 | 64 | 31.85 | 0.0283 |
| **64** | **8** | **32.74** | **0.0250** |

**Reconstruction fidelity (Table 1):** SLAT outperforms all baselines on both appearance and geometry metrics.

| Method | PSNR↑ | LPIPS↓ | CD↓ | F-score↑ | PSNR-N↑ | LPIPS-N↓ |
|--------|-------|--------|-----|----------|---------|----------|
| LN3Diff | 26.44 | 0.076 | 0.0299 | 0.9649 | 27.10 | 0.094 |
| 3DTopia-XL | 25.34 | 0.074 | 0.0128 | 0.9939 | 31.87 | 0.080 |
| CLAY | — | — | 0.0124 | 0.9976 | 35.35 | 0.035 |
| **Ours** | **32.74** | **0.025** | **0.0083** | **0.9999** | **36.11** | **0.024** |

---

## 11. 3D Editing Capabilities

**Detail variation:** Keep structure {pᵢ}, re-run stage 2 generation with different text prompts → same coarse shape, different appearance/detail.

**Region-specific editing:** Adapted from Repaint. Given a bounding box over voxels to edit:
1. Stage 1: generate new structures within bounding box, conditioned on unchanged areas + prompts
2. Stage 2: generate coherent latents for the edited region
Supports deletion, addition, and replacement of local regions.

---

## 12. Limitations

1. Two-stage pipeline (structure then latents) is less efficient than single-stage end-to-end methods
2. Image-to-3D does not separate lighting — shading/highlights from reference image are baked in. Could be addressed with lighting augmentation + PBR material prediction.