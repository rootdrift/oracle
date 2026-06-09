# Results

Four learners were evaluated on an identical, stratified 70/20/10 split of RSI-CB256: a custom
CNN trained from scratch (**TerraCNN**), a **ResNet-18** transfer-learning model, an **SVM**, and
a **Random Forest**. Architecture detail for TerraCNN is in
[model-architecture.md](model-architecture.md).

## Dataset — RSI-CB256

| Property | Value |
|----------|-------|
| Total images | 5,631 |
| Classes | 4 (Forest, Water, Cloudy, Desert) |
| Class balance | Forest / Water / Cloudy = 1,500 each; Desert = 1,131 (≈25% inter-class imbalance) |
| Split | Stratified 70/20/10 — train 3,941 / val 1,126 / test 563 (identical across all models) |
| Resolution | 256 px RGB scenes; resized to 128 × 128 for TerraCNN and classical learners |

Photometric notes: Desert averages 0.32 brightness, Water reaches 0.58; HSV saturation means
span 0.12–0.35 — diversity that rules out simple colour-histogram classifiers.

## Final metrics

| Model | Architecture | Test Accuracy | F₁_macro |
|-------|--------------|---------------|----------|
| ResNet-18 (transfer learning) | Pretrained 18-layer residual CNN, fine-tuned | **99.11%** | **0.9916** |
| TerraCNN (from scratch) | Custom 3-stage CNN (32→64→128) + GAP + 256-dense | 93.97% | 0.9390 |
| Random Forest | 100-tree ensemble, class-balanced, 49,152-D pixel vectors | — | ~0.94 |
| SVM (RBF) | RBF-kernel SVM, 49,152-D pixel vectors | — | ~0.90 |

> **Note on attribution.** The source experiment ran two CNN variants — `scratch` (TerraCNN) and
> `pretrained` (ResNet-18 transfer learning) — and selected the higher macro-F₁ model for the
> downstream latent analysis. The run logs record TerraCNN-from-scratch at 93.97% / 0.9390 and
> the pretrained ResNet-18 at 99.11% / 0.9916. The 99.11% figure therefore belongs to the
> transfer-learning model. Classical-learner accuracy was not separately reported; their
> macro-F₁ scores (RF ~0.94, SVM ~0.90) are taken from the comparative classification report.

The best learner overall was the **pretrained ResNet-18** (99.11% accuracy, 0.9916 macro-F₁),
confirming that ImageNet features transfer effectively to a constrained four-class remote-sensing
task. TerraCNN reached 0.9390 from scratch — within a point of Random Forest and ~5 points behind
the pretrained network, the expected gap for a small custom network trained on ~3,900 images.

## Error structure

| Model | Dominant confusion | Water error rate |
|-------|--------------------|------------------|
| ResNet-18 (best CNN) | 5 Water → Forest | 3.6% |
| Random Forest | Water → Forest | 8.1% |
| SVM | Water ↔ Forest | 17.4% |

All models struggle most with Water–Forest confusion, attributable to spectral overlap in
low-illumination scenes. The best CNN reduces this to 3.6% through learned spatial context; SVM
at 17.4% shows the limit of raw pixel vectors without hierarchical feature extraction.

## Latent-space analysis (ARI)

Penultimate 128-D activations from the CNN were reduced with PCA and projected with t-SNE:

| Step | Setting |
|------|---------|
| PCA | 200 components, retaining 96.0% variance |
| t-SNE | 2-D projection, perplexity = 30 |
| Cluster agreement | **Adjusted Rand Index (ARI) = 0.6478** |

ARI corrects cluster–label agreement for chance; 0.6478 indicates strong alignment between the
CNN's internal representation and the true class boundary even without labels at projection time.
Four coherent clusters emerge, with residual Water–Cloudy overlap accounting for the gap from a
theoretical maximum of 1.0.

## Takeaway

Transfer learning dominates on this constrained dataset, but the from-scratch TerraCNN remains
competitive with classical baselines while exposing an interpretable 128-D latent space — useful
when explainability matters more than the last point of accuracy.
