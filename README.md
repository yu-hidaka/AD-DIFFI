# AD-DIFFI: Adjusted Depth-based Isolation Forest Feature Importance for Robust Feature Ranking on Mixed-type Data

Official implementation of the method described in the manuscript:  
**"Adjusted Depth-based Isolation Forest Feature Importance for Robust Feature Ranking on Mixed-type Data"**

## Overview

AD-DIFFI (Adjusted Depth-based Isolation Forest Feature Importance) is a feature importance framework for Isolation Forest designed for **mixed-type medical and biological datasets**. The method extends the original DIFFI framework to address feature-type asymmetry between continuous and binary variables, which can distort feature rankings and reduce interpretability in heterogeneous anomaly detection settings.

AD-DIFFI addresses these issues through two key components:

- **Root-Split-Only (RSO):** Restricts the contribution of binary features to root splits, capturing their global separation role while reducing instability from deeper nodes.
- **Noise-based Z-score normalization:** Standardizes raw feature scores against type-specific null noise references, improving comparability across feature types and datasets.

This design improves signal-noise separation and feature-type fairness while preserving the structural interpretability of Isolation Forest.

## Manuscript Information

This repository is aligned with the latest manuscript version:

**Adjusted Depth-based Isolation Forest Feature Importance for Robust Feature Ranking on Mixed-type Data**

**Authors**
- Yu Hidaka
- Toru Imai
- Katsuhiro Omae

The manuscript proposes AD-DIFFI as an extension of DIFFI for anomaly detection with mixed continuous and binary features, and validates the method using simulation studies and multiple clinical benchmark datasets.

## Repository Structure

```text
AD-DIFFI/
├── src/
│   └── ad_diffi/
│       ├── __init__.py
│       └── core.py
├── notebooks/
│   ├── 01_Simulation_chapter3_bias_identification.ipynb
│   ├── 02_Simulation_chapter4_lambda_analysis.ipynb
│   ├── 03_Simulation_chapter5_rso_zscore_validation.ipynb
│   ├── 04_Real_World_Analysis_Annthyroid.ipynb
│   ├── 05_Real_World_Analysis_Stroke.ipynb
│   ├── 06_Real_World_Analysis_Breast_cancer.ipynb
│   ├── 07_Real_World_Analysis_Thyroid_subset.ipynb
│   └── 08_Real_World_Analysis_Hepatitis.ipynb
├── requirements.txt
└── README.md
```

## Installation

This project requires Python 3.12 or later.

Clone the repository from the main branch and install the dependencies:

```bash
git clone [https://github.com/yu-hidaka/AD-DIFFI.git](https://github.com/yu-hidaka/AD-DIFFI.git)
cd AD-DIFFI
pip install -r requirements.txt
```

## Usage

To reproduce the analyses, run the notebooks included in this repository.

### Environment setup

Add the following lines to the first cell of your notebook:

```python
import sys
import os

sys.path.append(os.getcwd())
```

### Example execution

```python
%run notebooks/07_Real_World_Analysis_Thyroid_subset.ipynb
```

### Kaggle-dependent analyses

For the Stroke (`05`) and Breast Cancer (`06`) notebooks, a Kaggle API token (`kaggle.json`) is required.

1. Upload `kaggle.json` when prompted.
2. The notebook will automatically configure the Kaggle API and download the required dataset.

## Experiments

This repository contains 8 primary notebooks corresponding to the simulation and benchmark analyses in the manuscript.

### Simulation studies

- **01_Simulation_chapter3_bias_identification.ipynb**  
  Evaluates feature-type asymmetry in the original DIFFI under null settings.

- **02_Simulation_chapter4_lambda_analysis.ipynb**  
  Studies the influence of the \(\lambda\) parameter on scoring behavior.

- **03_Simulation_chapter5_rso_zscore_validation.ipynb**  
  Validates the combined effect of RSO constraints and noise-based Z-score normalization.

### Real-data analyses

- **04_Real_World_Analysis_Annthyroid.ipynb**
- **05_Real_World_Analysis_Stroke.ipynb**
- **06_Real_World_Analysis_Breast_cancer.ipynb**
- **07_Real_World_Analysis_Thyroid_subset.ipynb**
- **08_Real_World_Analysis_Hepatitis.ipynb**

These notebooks reproduce the benchmark analyses described in the manuscript across multiple clinical mixed-type datasets.

## Interpretation of Scores

AD-DIFFI scores are standardized against type-specific null noise references.

- Higher scores indicate stronger contribution relative to the null baseline.
- Scores near the null reference indicate weak or noise-like contribution.
- Score interpretation should follow the current implementation and manuscript definition used in this repository.

## Reproducibility

The repository provides the Python implementation and experimental notebooks corresponding to the manuscript analyses. The associated code repository is:

[https://github.com/yu-hidaka/AD-DIFFI](https://github.com/yu-hidaka/AD-DIFFI)

For consistency, the public release should use **main** as the canonical branch.

## Citation

If you use this repository, please cite the manuscript:

**Yu Hidaka, Toru Imai, Katsuhiro Omae.**  
*Adjusted Depth-based Isolation Forest Feature Importance for Robust Feature Ranking on Mixed-type Data.*  
Preprint, 2026.
