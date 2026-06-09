# TerraCNN — Model Architecture

TerraCNN is a purpose-built convolutional network trained from scratch, sized for the RSI-CB256
dataset's scale (≈3,900 training images) and four-class structure. It is benchmarked against a
ResNet-18 transfer-learning model and two classical learners; see [results.md](results.md).

## Input specification

| Property | Value |
|----------|-------|
| Input tensor | 128 × 128 × 3 (RGB) |
| Normalisation | ImageNet channel statistics — μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225] |
| Why ImageNet stats | Aligns the scratch model's input distribution with the pretrained ResNet-18 baseline for a fair comparison |

## Architecture

```
Input: 128 × 128 × 3 (RGB, ImageNet-normalised)

Conv Block 1:  32 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2
Conv Block 2:  64 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2
Conv Block 3: 128 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2

Global Average Pooling           → 128-D feature vector

Dense: 256 units → ReLU → Dropout (p = 0.50)
Output: 4 units → Softmax
```

### Stage-by-stage

| Stage | Detail |
|-------|--------|
| Conv stage 1 | 32 filters, 3×3, ReLU, batch normalisation, 2×2 max-pool |
| Conv stage 2 | 64 filters, 3×3, ReLU, batch normalisation, 2×2 max-pool |
| Conv stage 3 | 128 filters, 3×3, ReLU, batch normalisation, 2×2 max-pool |
| Global Average Pooling | Collapses spatial dimensions → 128-D penultimate vector (used for latent analysis) |
| Dense head | 256 units, ReLU, dropout p = 0.50 |
| Classifier | 4 units, softmax |

The three-stage 32→64→128 widening with GAP (rather than large flattened dense layers) keeps the
parameter count modest and regularises the network for a moderate-sized remote-sensing dataset.

## Regularisation

| Technique | Setting |
|-----------|---------|
| Dropout | p = 0.50 (dense head) |
| L₂ weight decay | λ = 10⁻⁴ |
| Batch normalisation | After every convolution |

## Imbalance handling

- **Class-weighted cross-entropy:** `L_CE = -(1/N) Σ w_c y_ic log(p_ic)` with inverse-frequency
  weights — counters Desert under-representation.
- **Inverse-frequency sampler:** each mini-batch is class-balanced regardless of dataset skew, so
  Desert receives proportionally more samples per epoch.

## Augmentation

Horizontal flipping, ±20° rotation, and colour jitter (brightness = contrast = saturation = 0.3).
The jitter policy was motivated by EDA showing HSV means differing by up to 0.23 across classes;
a colour-insensitive policy would have leaked class information through photometric artefacts.

## Optimisation schedule

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (β₁ = 0.9, β₂ = 0.999) |
| Learning rate | 10⁻³ decaying to 2.5 × 10⁻⁴ by epoch 70 |
| Early stopping | patience = 10, monitoring validation loss |
| Max epochs | 100 (scratch run) |

## Latent-space probe

The 128-D Global-Average-Pooling vector is the feature representation used in the unsupervised
analysis (PCA → t-SNE → ARI) reported in [results.md](results.md).
