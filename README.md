# oracle

Applied machine learning research: land-cover classification from satellite imagery using a
purpose-built convolutional neural network benchmarked against classical learners. Framed as
security-relevant ML methodology — the same techniques (feature engineering, imbalance handling,
model selection, latent space analysis) transfer directly to threat detection pipelines.

---

## Overview

This project benchmarks four classification approaches on the RSI-CB256 satellite imagery
corpus: a custom convolutional neural network trained from scratch (**TerraCNN**), a
**ResNet-18** transfer-learning baseline, a support-vector machine, and a random forest
ensemble. The primary evaluation metric is macro-averaged F₁, which gives equal weight to
all four land-cover classes regardless of frequency — a deliberate choice that mirrors the
threat-detection requirement where rare events (attacks) must not be drowned by
majority-class accuracy.

The security-relevance is methodological. The techniques applied here — handling class
imbalance, tuning for minority-class recall, inspecting latent feature space, and comparing
parametric vs. non-parametric learners — are the same techniques used in intrusion detection,
malware classification, and anomaly detection pipelines. This project demonstrates those
skills in a fully reproducible, quantitatively rigorous setting.

---

## Objectives

- Establish whether a purpose-built shallow CNN can match or exceed transfer-learning baselines
  on a constrained remote-sensing dataset
- Quantify the performance gap between deep and classical learners under controlled, identical
  data splits
- Demonstrate imbalance mitigation strategies that preserve minority-class recall without
  over-fitting the majority
- Inspect latent representations learned by the CNN to verify genuine feature separation, not
  spurious correlation

---

## Dataset

**RSI-CB256** — Remote Sensing Image Classification Benchmark, 256px resolution RGB scenes.

| Class | Samples | Characteristics |
|-------|---------|-----------------|
| Forest | 1,500 | High saturation variance; green-dominant |
| Water | 1,500 | High brightness; spectrally distinct |
| Cloudy | 1,500 | Low saturation; significant illumination spread |
| Desert | 1,131 | Low saturation; brown-dominant; under-represented |

Total: **5,631 images** — approximately 25% inter-class imbalance (Desert vs. the three majority
classes). Desert averages 0.32 brightness; Water reaches 0.58. HSV saturation means span 0.12
to 0.35 — significant photometric diversity that rules out simple colour-histogram classifiers.

**Data split:** Stratified 70 / 20 / 10 (train 3,941 / val 1,126 / test 563), identical across
all three models to ensure comparability.

---

## Methodology

### TerraCNN Architecture

Purpose-built convolutional network designed for this dataset's scale and class structure.

```
Input: 128 × 128 × 3 (RGB, ImageNet-normalised)

Conv Block 1:  32 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2
Conv Block 2:  64 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2
Conv Block 3: 128 filters, 3×3 kernel → ReLU → BatchNorm → MaxPool 2×2

Global Average Pooling

Dense:    256 units → ReLU → Dropout (p = 0.50)
Output:     4 units → Softmax
```

Regularisation: dropout 0.50, L₂ weight decay λ = 10⁻⁴.

**Imbalance handling:**
- Class-weighted cross-entropy: `L_CE = -(1/N) Σ w_c y_ic log(p_ic)` — inverse-frequency
  weights assigned per class
- Inverse-frequency sampler: each mini-batch is class-balanced regardless of dataset skew;
  Desert receives proportionally more samples per epoch

**Augmentation:** horizontal flipping, ±20° rotation, colour jitter (brightness = contrast =
saturation = 0.3) — the jitter parameter was motivated by EDA showing HSV means differing by
up to 0.23 across classes; a colour-insensitive augmentation policy would have leaked class
information through photometric artefacts.

**Optimiser:** Adam with β₁ = 0.9, β₂ = 0.999. Learning rate decayed from 10⁻³ to 2.5 × 10⁻⁴
by epoch 70. Early stopping: patience = 10, monitoring validation loss.

**Normalisation:** ImageNet channel statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224,
0.225]) — applied to align with the pretrained ResNet-18 baseline and allow fair comparison.

### Support-Vector Machine

- Feature representation: images resized to 128 × 128 and flattened to 49,152-dimensional
  vectors; z-score normalised with `StandardScaler`
- Kernel: RBF — `K(x_i, x_j) = exp(−γ ‖x_i − x_j‖²)`
- Hyperparameter search: grid search over C ∈ {1, 10, 100} × γ ∈ {10⁻³, 10⁻², 10⁻¹},
  five-fold cross-validation on the training split
- No handcrafted features: raw pixel vectors only, matching TerraCNN's baseline condition

### Random Forest

- 100 estimators; `max_depth` and `n_estimators` selected by grid search over
  {100, 200} × {None, 20}
- Class-balanced sampling on each tree
- Out-of-bag estimates used as additional overfitting guard
- Same 49,152-dimensional z-score normalised vectors as SVM input

### Latent Space Analysis — PCA + t-SNE

Penultimate layer activations (128-dimensional) extracted from TerraCNN on the test split.

- PCA: reduced to 200 components, retaining 96.0% of variance
- t-SNE: two-dimensional projection, perplexity = 30
- Cluster quality measured with Adjusted Rand Index (ARI) against true class labels

ARI quantifies agreement between the discovered cluster structure and ground-truth labels,
corrected for chance — a value near 1.0 means the CNN's internal representations align
with the class boundary even without access to the labels during projection.

---

## Results

### Final Metrics

All four learners were evaluated on the same held-out 563-image test split.

| Model | Architecture | Test Accuracy | F₁_macro |
|-------|--------------|---------------|----------|
| ResNet-18 (transfer learning) | Pretrained 18-layer residual CNN, fine-tuned on RSI-CB256 | **99.11%** | **0.9916** |
| TerraCNN (from scratch) | Custom 3-stage CNN (32→64→128 filters) + GAP + 256-unit dense | 93.97% | 0.9390 |
| Random Forest | 100-tree ensemble, class-balanced, 49,152-D pixel vectors | — | ~0.94 |
| SVM (RBF) | RBF-kernel SVM, 49,152-D pixel vectors | — | ~0.90 |

The **ResNet-18 transfer-learning** model was the strongest learner overall (99.11% accuracy,
0.9916 F₁_macro), confirming that ImageNet-pretrained spatial features transfer effectively to
a constrained four-class remote-sensing task. The purpose-built **TerraCNN** reached 0.9390
F₁_macro from scratch — within a point of the Random Forest baseline (~0.94) and roughly five
points behind the pretrained network, the expected gap when a small custom network is trained
on ~3,900 images without pretraining. Accuracy for the classical learners was not separately
reported in the source experiment; their macro-F₁ scores (RF ~0.94, SVM ~0.90) are taken from
the comparative classification-report table. The pretrained ResNet-18 was selected as the
best-run model for the downstream latent-space analysis.

### Error Structure

| Model | Dominant confusion | Water error rate |
|-------|--------------------|-----------------|
| ResNet-18 (best CNN) | 5 Water → Forest | 3.6% |
| Random Forest | Water → Forest | 8.1% |
| SVM | Water ↔ Forest | 17.4% |

All models struggle most with Water–Forest confusion, likely due to spectral overlap in
low-illumination scenes. The best CNN (pretrained ResNet-18) reduces this error to 3.6% through
learned spatial context; SVM at 17.4% demonstrates the limitation of raw pixel vectors without
hierarchical feature extraction.

### Latent Space — ARI

**ARI = 0.6478** — strong agreement between TerraCNN's internal cluster structure and true
class labels. Four coherent clusters emerge in the t-SNE projection, with residual Water–Cloudy
overlap accounting for the gap from a theoretical maximum of 1.0.

PCA component retention: 200 components capture 96.0% of the variance in the 128-dimensional
penultimate representation.

---

## Security Relevance

The techniques applied here map directly to security detection problems:

| ML technique | Security application |
|--------------|----------------------|
| Macro F₁ as primary metric | Rare-class recall — attacks are the minority class in any production environment |
| Class-weighted loss + inverse-frequency sampler | Prevents benign-class dominance in intrusion detection |
| SVM with RBF kernel | Classical anomaly scoring; used in network intrusion detection (NSL-KDD, CICIDS) |
| Random Forest | Feature importance ranking in malware classification; explainability for SOC analysts |
| CNN spatial feature extraction | Network traffic image encoding; binary visualisation for malware |
| Latent space ARI analysis | Cluster validation in unsupervised threat grouping |
| Early stopping + dropout | Generalisation to novel attack variants; avoids over-fitting to training-set attack signatures |

The imbalance problem is especially relevant: a production IDS where 0.01% of traffic is
malicious must not achieve 99.99% accuracy by labelling everything benign. Macro F₁ and
class-weighted loss address this directly, and are standard practice in applied threat detection.

---

## Skills Demonstrated

| Skill area | Evidence |
|------------|---------|
| Deep learning | Custom CNN architecture design; PyTorch training pipeline; Adam with scheduling |
| Classical ML | SVM grid search; Random Forest with OOB validation; five-fold CV |
| Imbalance handling | Class-weighted cross-entropy; inverse-frequency mini-batch sampling |
| Feature engineering | Raw pixel vectors for classical models; augmentation policy grounded in EDA |
| Evaluation methodology | Macro F₁; stratified splits; controlled comparability across models |
| Latent space analysis | PCA dimensionality reduction (200 components, 96% variance); t-SNE; ARI |
| Security-relevant framing | Imbalance techniques directly applicable to IDS and malware classification |
| Python / ML tooling | PyTorch, torchvision, scikit-learn, NumPy, PCA, t-SNE |
| Reproducibility | Fixed seeds; stratified splits shared across all models; documented hyperparameters |
