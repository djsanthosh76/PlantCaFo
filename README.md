# PlantCaFo: Few-Shot Plant Disease Recognition
## Using Foundation Models

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)]()
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-green)]()

## Project Overview

Implementation and extension of PlantCaFo
(Jiang et al., Plant Phenomics 2025) for few-shot
plant disease recognition using foundation models.

**Achieves 84.46% on PlantVillage 38-way 1-shot**
(vs 62.53% in paper) and **85.33% on 4-shot**
(vs 82.63% in paper).

## Novel Contributions

1. **Multi-Scale Disease-Aware Prompting**
   Three semantic prompt levels per disease class
   (plant → disease → symptom) categorized by
   disease mechanism (fungal/bacterial/viral)

2. **Confidence-Based Support Selection**
   Filters support images using CLIP zero-shot
   confidence scores — top 50% most representative
   images per class used for cache model

## Pipeline Components

- CLIP ViT-B/32 (512-dim) — frozen
- DINOv2 ViT-B/14 (768-dim) — frozen
- Feature Combination (1280-dim)
- Weight Decomposition Matrix (WDM)
- Dilated Contextual Adapter (DCon-Adapter)
- Mixup + CutMix Augmentation
- Trained Cache Model

## Datasets

| Dataset | Images | Classes | Setting |
|---------|--------|---------|---------|
| PlantVillage | 52,222 | 38 | Lab |
| Cassava | 5,656 | 5 | Field |

> Datasets not included due to size.
> Download from:
> - PlantVillage: https://plantvillage.psu.edu/
> - Cassava: https://www.kaggle.com/c/cassava-disease

## Results

### PlantVillage 38-way K-shot

| K-shot | Ours | Paper PlantCaFo |
|--------|------|-----------------|
| 1-shot | 84.46% | 62.53% ✅ |
| 2-shot | 83.95% | 72.58% ✅ |
| 4-shot | 85.33% | 82.63% ✅ |
| 8-shot | 84.27% | 89.21% |
| 16-shot| 84.38% | 93.53% |

### Cassava 5-way K-shot

| K-shot | Ours | Paper PlantCaFo |
|--------|------|-----------------|
| 1-shot | 70.49% | 46.20% ✅ |
| 2-shot | 69.38% | 46.70% ✅ |
| 4-shot | 65.83% | 52.80% ✅ |
| 8-shot | 67.40% | 64.40% ✅ |

## Setup and Installation

```bash
# Install dependencies
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision
pip install numpy scikit-learn matplotlib seaborn
```

## How to Run

Mount Google Drive in Colab
Download PlantVillage and Cassava datasets
Run Cell 1-2 (setup and mount)
Run Cells 5-6 (create splits) — once only
Run Cells 9-10 (CLIP extraction) — once only
Run Cells A-C (DINO2 extraction) — once only
Run remaining cells for training and evaluation


> **Note:** Feature extraction files (.npy) are not
> included. Re-extract using the provided notebook.
> pv_splits.json and cas_splits.json are included
> for reproducibility.

## Project Structure
PlantCaFo_GitHub/
├── PlantCaFo_Main.ipynb    ← Main notebook
├── cache/
│   ├── pv_splits.json      ← Train/val/test splits
│   └── cas_splits.json     ← Train/val/test splits
├── results/
│   ├── accuracy_vs_shots.png
│   └── confusion_matrix.png
├── .gitignore
└── README.md

## Reference

X. Jiang et al., "PlantCaFo: An efficient few-shot
plant disease recognition method based on foundation
models," Plant Phenomics, vol. 7, p. 100024, 2025.
