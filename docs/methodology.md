# Methodology

## Motivation and context
This work builds on recent advances in high-frequency limit order book modeling that demonstrate the effectiveness of deep learning approaches operating directly on raw LOB states. In particular, CNN-based models such as DeepLOB highlighted the importance of capturing local microstructural interactions within the order book, while attention-based architectures like TransLOB showed how longer-range temporal dependencies can be modeled more effectively.

A major source of inspiration for this study is LIT, whose design explicitly focuses on jointly learning spatial and temporal interactions in the limit order book through structured attention mechanisms. These works motivated the decision to treat the LOB as a dynamic object in its raw form, rather than relying on hand-crafted features, in order to better capture the non-linear, non-stationary, and regime-dependent behavior characteristic of high-frequency market microstructure.

Accordingly, this project adopts a modern end-to-end pipeline inspired by LIT, DeepLOB, and TransLOB, and applies it to a public benchmark dataset (FI-2010). The objective is not to claim state-of-the-art performance, but to provide a clean, reproducible study with clearly motivated design choices and systematic analysis.

## What we implemented and why
- **End-to-end pipeline**: data -> labels -> model -> evaluation -> figures, to make experiments reproducible and auditable.
- **Custom directional labeling**: mid-price horizons with a tick-based neutral band to reduce noise and control class balance.
- **LIT-like patching + transformer encoder**: attention over spatio-temporal patches to capture longer dependencies than pure CNNs.
- **OvR loss with configurable class weights**: explicit control of down/flat/up trade-offs, enabling targeted improvements.
- **Per-class evaluation**: confusion matrix and per-class precision/recall/F1 to surface weaknesses (notably "up").

## Labels
- Labels are derived from mid-price changes at multiple horizons.
- A neutral band in ticks is used to reduce noise.

## Evaluation
- Primary metrics: accuracy and macro-F1.
- Confusion matrix saved for each run.

## Known limitation
- The "up" class is harder to recall with short training runs and conservative labeling.

## Planned improvements
- Tune neutral band and class weights to rebalance up/flat trade-offs.
- Explore decision thresholds for OvR logits.
