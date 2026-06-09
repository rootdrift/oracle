# Security Relevance

This project is land-cover classification on the surface, but every technique used maps directly
onto a security-detection problem. The relevance is **methodological** — the same modelling
decisions recur in intrusion detection, malware classification, and anomaly detection.

## Technique-to-security mapping

| ML technique (this project) | Security application |
|------------------------------|----------------------|
| Macro-F₁ as primary metric | Rare-class recall — attacks are the minority class in any production environment; accuracy alone hides missed detections |
| Class-weighted loss + inverse-frequency sampler | Prevents benign-class dominance in intrusion detection where malicious traffic is a tiny fraction |
| CNN spatial feature extraction | Network-traffic image encoding and binary-to-image malware visualisation, where spatial structure carries signal |
| SVM with RBF kernel | Classical anomaly scoring; used in network intrusion detection (e.g. NSL-KDD, CICIDS) |
| Random Forest | Feature-importance ranking for malware classification; explainable scores for SOC analysts |
| Transfer learning (ResNet-18) | Reusing pretrained representations when labelled attack data is scarce — common in threat detection |
| Latent-space ARI analysis | Cluster validation for unsupervised threat grouping and campaign attribution |
| Early stopping + dropout | Generalisation to novel attack variants; avoids over-fitting to known training-set signatures |

## The imbalance problem as a security primitive

The central methodological point transfers cleanly: a production IDS where 0.01% of traffic is
malicious must **not** achieve 99.99% accuracy by labelling everything benign. The mitigations
demonstrated here — macro-F₁ as the headline metric, class-weighted cross-entropy, and
inverse-frequency sampling — are standard practice in applied threat detection precisely because
they protect minority-class (attack) recall.

## Mapping the three approaches to detection roles

### Image classification → malware visualisation detection
Binaries rendered as greyscale/RGB images expose byte-level structure; a CNN like the TerraCNN
built here learns hierarchical spatial features that distinguish families — the same pipeline
shape used in vision-based malware classifiers.

### SVM → network intrusion detection
RBF-kernel SVMs on engineered flow features are a long-standing IDS baseline. The grid-search +
five-fold CV discipline shown here is exactly how such detectors are tuned without leaking the
test distribution.

### Random Forest → threat classification
Tree ensembles give per-feature importance, which SOC analysts can read to justify a verdict —
valuable where explainability and auditability are requirements (e.g. cleared environments), not
optional extras.

## Transferable skills for security ML

- Choosing metrics that survive class imbalance.
- Tuning for minority-class recall without over-fitting the majority.
- Comparing parametric (SVM), non-parametric (RF), and deep (CNN/transfer) learners under
  identical, leakage-free splits.
- Inspecting latent representations to confirm a model learned signal, not spurious correlation —
  the same validation an analyst needs before trusting an automated detector.
