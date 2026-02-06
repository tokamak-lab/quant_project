# LIT-like LOB Pipeline (FI-2010)

End-to-end ML/Quant pipeline for limit order book (LOB) prediction using a LIT-like patching + attention architecture.

## What this project adds
- End-to-end pipeline: data -> labels -> model -> evaluation -> figures.
- Custom directional labeling on FI-2010 with a configurable neutral band.
- Rigorous evaluation: macro-F1, confusion matrix, ROC curves, calibration.

## Inspirations
This work is inspired by LIT, DeepLOB, and TransLOB. The goal is not to reproduce their results, but to build a clean, reproducible pipeline and evaluate it on public data.

## Data
- Public dataset: FI-2010 (benchmark for LOB prediction).
- No proprietary data is stored in the repo.

See data/README.md for instructions.

## Quickstart
1) Install dependencies

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Download FI-2010 (see data/README.md)

3) Train a LIT-like model

```
python -m src.train --config configs/lit_like.yaml
```

## Repo structure
- src/: data processing, labels, models, training, evaluation
- configs/: experiment configs
- results/: figures and metrics
- notebooks/: optional analysis notebooks

## Notes
- Labels are generated from mid-price changes with a tick-based neutral band.
- FI-2010 feature ordering can vary; configure it in configs/lit_like.yaml.

## Limitations and next steps
- Current weakness: class "up" recall is lower than down/flat.
- Next steps: tune neutral band, class weights, and decision threshold for a better trade-off.
- Add longer runs and learning-rate scheduling once the quick iteration loop is stable.
