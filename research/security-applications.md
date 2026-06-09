# Security Applications

This document expands the security relevance of the project into three concrete application areas.
It complements [security-relevance.md](security-relevance.md), which gives the technique-to-security
mapping table; here each mapping is developed in depth with the specific transfer mechanism.

The thesis throughout: land-cover classification is the *vehicle*, but every modelling decision —
spatial feature learning, multi-class output, and imbalance handling — is a transferable security
primitive. The figures referenced (TerraCNN 93.97% from scratch; pretrained ResNet-18 99.11%) are
detailed in [results.md](results.md).

## 1. CNN image classification → malware visualisation detection

**Technique in this project:** TerraCNN learns hierarchical spatial features (32→64→128 filters
across three convolutional stages) directly from raw RGB pixels, with no handcrafted features.

**Security transfer:** malware visualisation re-casts a binary as an image — each byte mapped to a
pixel intensity, the file laid out as a 2-D greyscale or RGB grid. Malware families share
byte-level structural motifs (packer stubs, section layouts, padding patterns) that appear as
**visual texture** in this representation. A convolutional network of exactly the shape built here
learns those textures as spatial features:

| TerraCNN element | Malware-image analogue |
|------------------|------------------------|
| Conv stages over RGB texture | Conv stages over byte-grid texture (section/opcode patterns) |
| Hierarchical 32→64→128 widening | Low-level byte motifs → mid-level structures → family-level signatures |
| Global Average Pooling → 128-D vector | Compact family embedding for similarity / clustering |
| Softmax over 4 classes | Softmax over malware families |

The advantage is the same one demonstrated here: the CNN reaches its result **without
signatures**, so it can generalise to packed or lightly mutated variants that defeat hash- and
string-based detection. The from-scratch result (93.97%) is the relevant evidence — it shows a
modest custom CNN extracts genuine structure from raw pixels, which is precisely the constraint in
vision-based malware detection where pretrained ImageNet features transfer less cleanly to
byte-grid images than they do to natural scenes.

## 2. Multi-class output → threat categorisation

**Technique in this project:** a four-class softmax (Forest / Water / Cloudy / Desert) with
macro-averaged F₁ as the headline metric, giving every class equal weight regardless of frequency.

**Security transfer:** real detection is rarely binary "malicious / benign". SOC pipelines need to
**categorise** — ransomware vs. trojan vs. dropper; reconnaissance vs. exploitation vs.
exfiltration; phishing vs. BEC vs. malware-delivery. The multi-class structure here maps directly:

- The softmax-over-N-classes head is identical in shape; only the label set changes.
- Macro-F₁ is the correct metric for threat categorisation because it refuses to let a common
  category (e.g. commodity adware) mask poor recall on a rare-but-severe one (e.g. targeted
  implant). A model that scores 99% accuracy by nailing the majority category while missing the
  dangerous minority is worthless to a defender — macro-F₁ exposes exactly that failure.
- The **error-structure analysis** demonstrated here (Water↔Forest confusion: 3.6% for the best
  CNN vs. 17.4% for the SVM) is the same confusion-matrix discipline a SOC uses to find which
  threat categories a classifier conflates — the categories most likely to produce a misrouted
  or under-prioritised incident.

The latent-space probe (PCA → t-SNE → ARI = 0.6478) extends this: it validates that the learned
classes form coherent clusters, the analogue of confirming that threat categories are genuinely
separable in the model's representation rather than overlapping into a single ambiguous blob.

## 3. Class-imbalance handling → rare attack classes

**Technique in this project:** Desert is under-represented (1,131 vs. 1,500 for each other class,
≈25% inter-class imbalance). Two mitigations preserve minority recall:

- **Class-weighted cross-entropy** — `L_CE = -(1/N) Σ w_c y_ic log(p_ic)` with inverse-frequency
  weights, so errors on the rare class cost more.
- **Inverse-frequency sampler** — each mini-batch is class-balanced, so the rare class is seen
  proportionally more often per epoch.

**Security transfer:** this is the single most directly transferable element, because security
datasets are *catastrophically* more imbalanced than this one. In production network traffic,
malicious events can be **well under 1%** — sometimes 0.01% — of all flows. The naïve failure mode
is the same one macro-F₁ is designed to catch: a model achieves 99.99% accuracy by labelling
everything benign and detecting nothing.

| Imbalance mechanism (this project) | Rare-attack-class application |
|------------------------------------|-------------------------------|
| Macro-F₁ as headline metric | Stops majority "benign" class hiding missed detections |
| Inverse-frequency class weights | Raises the cost of missing rare attack events (e.g. novel C2 beacons) |
| Inverse-frequency mini-batch sampler | Ensures rare attack signatures appear in training despite scarcity |
| Early stopping + dropout | Generalisation to *unseen* attack variants, not memorised signatures |

The Desert case is a scaled-down rehearsal of the rare-attack-class problem: a defender training a
detector on highly imbalanced security data uses exactly these techniques to keep recall on the
rare, high-impact class from collapsing. Demonstrating them on a clean, reproducible benchmark
proves the methodology before it is applied to noisier, higher-stakes security data.

## Why this matters for cleared detection work

Explainability and auditability are first-order requirements in cleared environments, not
optional extras. Two project elements speak to this directly:

- **Random Forest feature importance** gives per-feature justification a human analyst can read —
  the basis for an auditable verdict.
- **Latent-space validation (ARI)** is the discipline of confirming a model learned signal rather
  than spurious correlation *before* it is trusted to triage — the same assurance step a detection
  engineer must perform before an automated classifier is allowed to action or de-prioritise an
  alert.
