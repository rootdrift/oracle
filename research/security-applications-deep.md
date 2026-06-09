# Security Applications — Deep Dive Reference

Extends [security-applications.md](security-applications.md) and
[security-relevance.md](security-relevance.md) with deeper technical background on four ML-security
areas the project's techniques map onto: malware visualisation, NIDS, imbalanced threat
classification, and adversarial robustness. Background document — cites the project's reported
figures, introduces no new ones.

---

## 1. Malware visualisation

### 1.1 How a binary becomes an image

The standard technique (popularised by Nataraj et al., 2011, with the *Malimg* dataset): read the
binary as a stream of bytes, map each byte (0–255) to a pixel intensity, and lay the stream out
row-by-row into a 2-D **greyscale** image (width often fixed by file size; height follows). Variants
encode three derived streams as RGB channels, or use entropy/structural encodings.

Why it works: malware of the same **family** shares structural motifs — packer stubs, section
boundaries, code vs data regions, padding — that appear as repeatable **visual textures** in the
byte image. `.text`, `.data`, `.rsrc` sections produce visually distinct bands; reused code yields
recurring patterns across samples.

### 1.2 What CNN classification achieves

A CNN learns those textures as spatial features (exactly TerraCNN's mechanism: hierarchical
32→64→128 filters over a grid — see
[cnn-architecture-background.md](cnn-architecture-background.md) §1). Advantages over signature/hash
detection:

- **No handcrafted signatures.** Generalises to **packed or lightly mutated** variants that defeat
  hash/string matching, because texture survives small mutations better than exact bytes.
- **Family-level embeddings.** A GAP embedding (like TerraCNN's 128-D vector) supports similarity
  search and clustering of related samples.
- **Static, fast.** Operates without execution (no sandbox detonation needed for the first pass).

Why the **from-scratch** result (93.97%) is the relevant evidence here: byte-grid images are **not**
natural photos, so ImageNet-pretrained features transfer *less* cleanly than they do to scenes — this
is a domain where a purpose-trained CNN (or domain-pretrained one) is often the right call, the
scenario flagged in [cnn-architecture-background.md](cnn-architecture-background.md) §2.3.

### 1.3 Real-world implementations (verify before citing)

- **Academic:** Nataraj et al. (2011) image-texture malware classification; the **Microsoft Malware
  Classification Challenge (BIG 2015)** on Kaggle drove much byte/asm-image CNN work; numerous
  follow-on CNN papers.
- **Industry:** major AV/EDR vendors (including Kaspersky and others) publicly describe using ML —
  including deep learning over binary representations — in their static-analysis stacks. **Treat
  specific product/architecture claims as requiring verification**; do not assert a named vendor uses
  a *specific* image-CNN pipeline without a source.

### 1.4 Limits (honest)

Adversaries can pad/reorder sections to alter texture (adversarial evasion, §4); heavy packing/
encryption can flatten structure into high-entropy noise; static-only views miss runtime behaviour.
Image-CNN detection is one layer, complementary to dynamic analysis.

---

## 2. Network intrusion detection (the SVM/RBF mapping)

### 2.1 How the project's SVM maps to NIDS

The project trains an **RBF-kernel SVM** as a comparison learner. In NIDS, RBF-SVMs on **engineered
flow features** are a long-standing baseline (e.g. on the classic **NSL-KDD** and **CICIDS2017**
research datasets). The mapping:

- **RBF kernel** implicitly projects features into a high-dimensional space where non-linearly
  separable classes (benign vs attack) can be split by a hyperplane — useful when attack/benign
  boundaries are non-linear in the raw feature space.
- The **grid-search + k-fold CV** discipline shown in the project (tuning C and γ without leaking
  the test split) is exactly how such detectors must be tuned — leakage here would massively
  overstate detection.
- The project's lesson that **raw pixel vectors limit the SVM** (17.4% Water↔Forest error vs 3.6%
  for the CNN, results.md) is the NIDS lesson too: **feature engineering dominates classical NIDS
  performance.** An SVM is only as good as its flow features.

### 2.2 What features work well in NIDS

Flow-level statistics: duration, bytes/packets per direction, inter-arrival times, flag counts,
flow rates, protocol, port behaviour, and windowed aggregates (connections per host, etc.).
Engineered/temporal features generally outperform raw payload for classical learners.

### 2.3 State of the art (brief, verify specifics)

Classical learners (SVM, **Random Forest**, gradient boosting) remain strong and *interpretable*
baselines on tabular flow data and are still competitive in many published comparisons. Deep methods
(autoencoders for anomaly detection, CNNs/RNNs over flow sequences, graph-based detection) lead on
some benchmarks but add cost and opacity. A recurring research caveat: results on aged datasets
(KDD'99/NSL-KDD) overstate real-world performance; concept drift and base-rate fallacy (§3) bite
hard in production. *Do not cite a specific SOTA accuracy without a source.*

---

## 3. Imbalanced threat classification

### 3.1 From Desert to rare attacks

The project's Desert class (≈25% under-represented) is a **mild, scaled-down rehearsal** of the
security reality: malicious events can be **well under 1%** (sometimes 0.01%) of traffic. The naïve
failure — 99.99% accuracy by labelling everything benign — is precisely why the project leads with
**macro-F₁**, **class-weighted cross-entropy**, and **inverse-frequency sampling**
([security-applications.md](security-applications.md) §3,
[cnn-architecture-background.md](cnn-architecture-background.md) §4.4).

### 3.2 SMOTE and other balancing approaches

| Approach | What it does | When to use / caution |
|----------|--------------|------------------------|
| **Class weighting** (used here) | Up-weights minority-class errors in the loss | Cheap, no data synthesis; first choice for deep nets and images |
| **Inverse-frequency / balanced sampling** (used here) | Rebalances each mini-batch | Good for deep learning; pairs with weighting |
| **Random oversampling / undersampling** | Duplicate minority / drop majority | Oversampling risks overfitting duplicates; undersampling discards data |
| **SMOTE** (Synthetic Minority Over-sampling) | Creates *synthetic* minority points by interpolating between a minority sample and its k nearest minority neighbours **in feature space** | Strong for **tabular** data (e.g. NIDS flow features). **Caution:** interpolating raw pixels/high-dim image data produces unrealistic samples — class weighting/focal loss is usually better for images |
| **Borderline-SMOTE / ADASYN** | Focus synthesis near the decision boundary / hard examples | Useful when the boundary is the problem; can amplify noise |
| **Focal loss** | Down-weights easy examples, focuses on hard/rare ones | Popular in extreme imbalance (detection) |

Crucial deployment caveat — **base-rate fallacy**: at a 0.01% attack rate, even a 99.9%-specific
detector floods analysts with false positives. Balancing techniques fix *training*; **operating-point
/ threshold tuning, precision-recall (not ROC) evaluation, and alert-cost modelling** fix
*deployment*. Macro-F₁ and PR-AUC are the honest metrics; raw accuracy and even ROC-AUC can mislead
under extreme imbalance.

---

## 4. Adversarial examples against ML security systems

### 4.1 How CNNs are fooled

An **adversarial example** is an input perturbed by a small, often human-imperceptible amount that
flips the model's prediction. Standard methods: **FGSM** (one gradient-sign step), **PGD** (iterative,
the standard strong white-box attack), C&W, and **black-box/transfer** attacks that craft examples on
a surrogate model and transfer them. They exploit that high-dimensional decision boundaries are
locally brittle — the model relies on features that are predictive but not robust.

### 4.2 Why this matters specifically for ML *security* systems

Unlike a land-cover classifier, a security model has a **motivated adversary** whose goal is evasion:

- **Malware-image evasion:** an attacker can append/reorder bytes, pad sections, or insert dead code
  to shift the image texture and flip a CNN's family/verdict **without changing functionality** —
  directly attacking the §1 pipeline.
- **NIDS evasion:** mimicry attacks shape malicious flows to resemble benign feature distributions
  (timing, sizes), defeating the §2 detector.
- **Phishing/NLP evasion:** paraphrase and homoglyph tricks (cf. the MIRAGE work on surface
  perturbation) evade content classifiers.

The defender's asymmetry: a vision benchmark's errors are random; a security model's errors are
**adversarially chosen** to be maximally harmful.

### 4.3 Implications for deploying ML in security

- **Assume evasion.** Evaluate against adaptive adversaries, not just a held-out test set. The
  project's clean-split rigour is necessary but **not sufficient** for a security deployment.
- **Defences are partial.** Adversarial training (train on PGD examples), input preprocessing,
  ensembling, and detection-of-adversarial-inputs all help but none fully solve it; robustness
  trades against clean accuracy.
- **Defence in depth.** ML is one layer; pair it with signatures, behavioural/dynamic analysis, and
  human review (the latent-space validation and RF feature-importance the project demonstrates feed
  that human-review/auditability requirement, [security-relevance.md](security-relevance.md)).
- **Explainability aids robustness triage.** Inspectable models (RF importances, GAP embeddings) let
  analysts spot when a verdict rests on fragile features — exactly the assurance step before trusting
  an automated detector in a cleared environment.

---

## 5. Cross-references and open items

- Technique→security mapping table: [security-relevance.md](security-relevance.md).
- In-depth per-application development: [security-applications.md](security-applications.md).
- CNN/transfer/GAP/imbalance theory: [cnn-architecture-background.md](cnn-architecture-background.md).
- [ ] Verify any named-vendor ML-pipeline claim before citing (§1.3).
- [ ] Verify any NIDS SOTA accuracy figure before citing (§2.3).

> No metric beyond those in the existing oracle documents is introduced. References (Nataraj et al.
> 2011; Microsoft Malware Classification Challenge 2015; NSL-KDD/CICIDS2017; SMOTE [Chawla et al.
> 2002]; FGSM [Goodfellow et al. 2015]; PGD [Madry et al. 2018]) are standard — confirm before formal
> citation.
