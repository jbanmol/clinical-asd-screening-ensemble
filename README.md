# Clinical ASD Screening Ensemble

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-02569B)
![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-FF6600)
![Healthcare AI](https://img.shields.io/badge/Healthcare_AI-Clinical_ML-0EA5E9)
![Reproducible](https://img.shields.io/badge/Reproducible-Inference-10B981)

Leakage-safe machine learning pipeline for ASD-vs-TD screening from behavioral touch-interaction data. The project converts raw coloring-session records into child-level features, trains calibrated ensembles, selects clinically constrained thresholds, and exports a reproducible inference bundle.

## What This Demonstrates

- Subject-level data splitting to avoid child/session leakage
- Feature engineering from raw behavioral interaction logs
- Calibrated ensembles with LightGBM, XGBoost, Balanced Random Forest, and ExtraTrees
- Sensitivity/specificity-aware threshold selection
- Reproducible prediction through saved preprocessors, models, schema, and thresholds
- End-to-end CLI for scoring new raw data

## Results Snapshot

| Metric | Approx. Value |
|---|---:|
| Holdout AUC | 0.91 |
| Sensitivity | 0.88 |
| Specificity | 0.80 |

These are experimental project results, not clinical claims. The main value is the careful ML workflow: grouping, calibration, threshold policy, and reproducible inference.

## System Flow

```text
Raw coloring sessions
  -> child-level feature extraction
  -> grouped train/holdout split
  -> group-aware cross-validation
  -> calibrated base models
  -> sensitivity/specificity sub-ensembles
  -> threshold selection and transfer
  -> exported model bundle
  -> end-to-end prediction CLI
```

## Quick Start

```bash
git clone https://github.com/jbanmol/clinical-asd-screening-ensemble.git
cd clinical-asd-screening-ensemble
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/python scripts/e2e_predict.py --raw data/raw/phase2_file_keys
```

## Key Files

| Path | Purpose |
|---|---|
| `scripts/clinical_fair_pipeline.py` | Training, grouped CV, calibration, ensembling, and threshold selection |
| `scripts/e2e_predict.py` | End-to-end prediction from raw sessions |
| `scripts/predict_cli.py` | Loads the exported model bundle and scores aligned features |
| `src/` | Reusable preprocessing, modeling, representation, and evaluation utilities |
| `references/results.md` | Detailed metrics and robustness notes |

## Tech Stack

Python, pandas, scikit-learn, LightGBM, XGBoost, imbalanced-learn, UMAP, joblib.

## Research Direction

This project reflects my interest in reliable AI for behavioral and clinical data: models that are not only accurate, but auditable, calibrated, and reproducible.
