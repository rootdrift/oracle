# CNN Architecture Background — Reference

Theory reference for the architecture decisions in [model-architecture.md](model-architecture.md)
and the comparison in [results.md](results.md). Explains CNN building blocks, the scratch-vs-transfer
gap, GAP-vs-flatten, and the dataset. Background document — it cites the engagement's reported
figures, introducing no new ones.

---

## 1. CNN building blocks

### 1.1 The convolution operation

A convolutional layer slides a small learnable **kernel** (here 3×3) across the input, computing a
dot product at each spatial position to produce a **feature map**. Key properties and why they
matter for image/grid data:

- **Local receptive field:** each output depends only on a small input neighbourhood — well-matched
  to images, where signal is local (edges, textures).
- **Parameter sharing:** the *same* kernel weights are reused across all positions, so the layer
  learns a feature detector that is **translation-equivariant** and uses vastly fewer parameters
  than a dense layer over the same input.
- **Stacking → hierarchy:** stacking conv layers grows the effective receptive field, so early
  layers learn low-level features (edges, colour blobs) and deeper layers compose them into
  mid-/high-level structures. TerraCNN's **32→64→128** widening is exactly this: more filters deeper
  in, capturing increasingly abstract patterns (model-architecture.md).
- **Channels:** a conv layer over RGB has kernels of depth 3; the number of *output* channels = the
  number of kernels (32, then 64, then 128 here).

### 1.2 Pooling

**Max-pooling** (2×2 here) downsamples each feature map by taking the maximum in each window. It:

- reduces spatial resolution (and compute) as depth increases,
- grows the receptive field, and
- gives a degree of **local translation invariance** (small shifts don't change the max much).

TerraCNN halves spatial dimensions after each of the three conv stages, so a 128×128 input is
reduced before the pooling-to-vector step (§3).

### 1.3 Batch normalisation

**BatchNorm** normalises each layer's pre-activations to roughly zero-mean/unit-variance per
mini-batch (then rescales with learnable γ, β). Benefits:

- **Stabilises and accelerates training** by reducing internal covariate shift / smoothing the loss
  landscape, allowing higher learning rates.
- Acts as a **mild regulariser** (batch statistics add noise).

TerraCNN places BatchNorm **after every convolution** (model-architecture.md), standard for a
from-scratch network and important for training a deep-ish stack from random init without divergence.

### 1.4 Dropout

**Dropout** randomly zeroes a fraction `p` of units during training (here `p = 0.50` on the dense
head), forcing the network not to rely on any single feature and approximating an ensemble. It is a
**regulariser** that combats overfitting — critical when training from scratch on only ~3,900 images
(results.md), where overfitting risk is high. At inference dropout is off (activations scaled
accordingly).

### 1.5 Activation and the classifier head

- **ReLU** (`max(0,x)`) provides non-linearity cheaply and avoids the vanishing-gradient problems of
  saturating activations.
- The **softmax** output (4 units) turns logits into a probability distribution over the four
  classes; trained with cross-entropy (class-weighted here for imbalance, model-architecture.md §
  Imbalance handling).

---

## 2. From-scratch (TerraCNN) vs transfer learning (ResNet-18)

### 2.1 The reported gap

Identical 70/20/10 split, four learners: **ResNet-18 transfer learning 99.11% / 0.9916 macro-F₁**
vs **TerraCNN from scratch 93.97% / 0.9390** (results.md). ~5 points. Why?

### 2.2 What transfer learning actually provides

ResNet-18 is **pretrained on ImageNet** (~1.2M images, 1000 classes). Fine-tuning it here means:

- **Reused low/mid-level features.** Edge/texture/shape detectors learned from a million images are
  already excellent and **generic** — they transfer to remote-sensing scenes without re-learning.
  The model only has to adapt higher layers + the new 4-class head.
- **A vastly larger effective training set.** The representation encodes knowledge from ImageNet, so
  the ~3,900 in-domain images are enough to specialise rather than learn vision from zero.
- **Better-conditioned starting point + residual connections.** ResNet's skip connections ease
  gradient flow in a deeper network than TerraCNN.
- **Regularisation by prior.** Pretrained weights are a strong prior that resists overfitting the
  small dataset.

TerraCNN, by contrast, must learn **everything from random init on ~3,900 images** — it cannot reach
the same feature quality from so little data. 93.97% from scratch is, as results.md notes, the
*expected* result: within a point of Random Forest and ~5 behind the pretrained net. The matched
ImageNet input normalisation (model-architecture.md) keeps the comparison fair.

### 2.3 When training from scratch is appropriate

Transfer learning is not always the answer. From-scratch is preferable when:

- **Domain mismatch:** the target images are unlike ImageNet's natural photos — e.g. **byte-grid
  malware images** (see [security-applications-deep.md](security-applications-deep.md) §1), medical
  scans, or spectrograms — where ImageNet features transfer poorly.
- **Architectural control / interpretability:** you need a small, inspectable network (TerraCNN's
  128-D GAP latent is used for the PCA→t-SNE→ARI probe, results.md) rather than a fixed pretrained
  backbone.
- **Constraints:** tight parameter/latency budgets, or licensing/provenance requirements that forbid
  third-party pretrained weights (relevant in cleared environments).
- **Sufficient in-domain data:** with enough labelled data, a custom network can match or beat a
  transferred one while being purpose-fit.

The project's value is precisely in running **both** under identical conditions: it quantifies the
transfer benefit *and* retains an interpretable scratch model.

---

## 3. Global Average Pooling vs flatten

The penultimate step can either **flatten** the final conv feature maps into a long vector feeding a
large dense layer, or apply **Global Average Pooling (GAP)** — averaging each feature map to a single
number, yielding one value per channel (here a **128-D** vector, one per final-stage filter).

| | Flatten → big dense | Global Average Pooling (TerraCNN) |
|---|---------------------|-----------------------------------|
| Parameters | **Many** — dense weights = (H·W·C)×units; dominates the parameter count | **Near zero** — averaging has no learnable params |
| Overfitting | High risk on small data (huge dense layer memorises) | **Lower** — drastic parameter reduction regularises |
| Spatial info | Preserves exact spatial layout | Discards fine spatial position (keeps "how much of feature C overall") |
| Input-size rigidity | Fixed by H·W | Tolerant to input size |
| Interpretability | Entangled | Each dim ≈ a channel's average activation — a clean **embedding** |

**Trade-offs and why GAP here:** GAP throws away fine spatial position, which can cost accuracy when
exact location matters. But for this dataset its advantages dominate: it **slashes parameters and
overfitting** on ~3,900 images, and it yields a compact, meaningful **128-D embedding** — exactly the
vector used for the latent-space analysis (PCA 200-comp → t-SNE → ARI = 0.6478, results.md). The
architecture note's rationale ("rather than large flattened dense layers… keeps the parameter count
modest and regularises") is the standard, correct reasoning; GAP is the modern default in compact
classifiers (popularised by Network-in-Network and used throughout ResNet-style heads).

---

## 4. The dataset

### 4.1 As used in this project

results.md records: **5,631 images, four classes (Forest, Water, Cloudy, Desert)**, balance
1,500/1,500/1,500/1,131 (Desert under-represented, ≈25% inter-class imbalance), 256 px RGB resized to
128×128, stratified 70/20/10 (train 3,941 / val 1,126 / test 563).

### 4.2 Naming caveat (verify)

> The canonical **RSI-CB256** benchmark in the literature is a *larger, multi-category* remote-sensing
> dataset (tens of fine-grained land-use categories at 256 px). The four-class Forest/Water/Cloudy/
> Desert collection used here matches the widely-circulated Kaggle "Satellite Image Classification"
> set. **Confirm the exact dataset provenance/name** before publishing — the *methodology* is
> unaffected, but the dataset label should be accurate. (Flagged, not asserted.)

### 4.3 Remote-sensing classification specifics

- **Spectral overlap:** the dominant error across all models is **Water↔Forest** confusion
  (results.md), attributed to spectral similarity in low-illumination scenes — a remote-sensing
  staple where RGB alone (no near-infrared band) limits separability. The best CNN cuts it to 3.6%
  via learned spatial context; the SVM on raw pixels sits at 17.4%.
- **Photometric variation** (brightness/HSV spread noted in both docs) motivated the colour-jitter
  augmentation policy and rules out naïve colour-histogram classifiers.

### 4.4 Why 25% class imbalance matters, and standard handling

Even a "mild" 25% under-representation of Desert biases an un-corrected model toward the larger
classes and depresses minority recall — and **accuracy hides it** (a model can score well while
quietly failing the rare class). TerraCNN's mitigations are the standard toolkit:

- **Macro-F₁ as the headline metric** (equal weight per class, exposes minority failure),
- **class-weighted cross-entropy** (inverse-frequency weights raise the cost of Desert errors),
- **inverse-frequency mini-batch sampling** (Desert seen proportionally more per epoch).

Other standard approaches not central here: oversampling/undersampling, **SMOTE**-style synthetic
oversampling (better suited to tabular than image data — see
[security-applications-deep.md](security-applications-deep.md) §3), and focal loss. This imbalance is
a scaled-down rehearsal of the rare-attack-class problem developed in the security-applications docs.

---

## 5. Cross-references and open items

- Architecture specifics: [model-architecture.md](model-architecture.md).
- Comparative metrics and latent analysis: [results.md](results.md).
- Security transfer of these techniques: [security-applications-deep.md](security-applications-deep.md),
  [security-relevance.md](security-relevance.md).
- [ ] Verify exact dataset provenance/name vs canonical RSI-CB256 (§4.2).

> No metric beyond those reported in the existing oracle documents is introduced. Architectural
> claims reflect standard CNN theory (LeCun-style convnets, BatchNorm [Ioffe & Szegedy 2015],
> Dropout [Srivastava et al. 2014], GAP [Lin et al. 2013], ResNet [He et al. 2015]); confirm
> citations before formal use.
