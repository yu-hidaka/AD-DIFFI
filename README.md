# AD-DIFFI: Adjusted DIFFI for Robust Feature Importance in Isolation Forest

Official implementation of the paper:  
**"Adjust DIFFI (AD-DIFFI): Robust Feature Importance for Mixed-Type Data in Isolation Forest"**

## Overview
AD-DIFFI (Adjusted Depth-based Isolation Forest Feature Importance) is a novel feature importance method specifically designed for **mixed-type medical and biological data**. It addresses the inherent biases in the original DIFFI method when handling datasets with both continuous and binary features.

### Key Features
- **Root-Split-Only (RSO)**: Corrects the overestimation of binary noise by focusing on early splits.
- **Z-score Normalization**: Standardizes importance scores using a noise-based baseline.
- **Reproducibility**: Automated data acquisition and preprocessing for all benchmark datasets.

---

## Repository Structure
```text
AD-DIFFI/
├── src/
│   └── ad_diffi/
│       ├── __init__.py
│       └── core.py         # Main AD-DIFFI implementation
├── notebooks/              # 8 primary experiment notebooks
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
This project requires Python 3.12 or later. To set up the environment, clone the repository and install the dependencies:

```bash
git clone [https://github.com/yu-hidaka/AD-DIFFI.git](https://github.com/yu-hidaka/AD-DIFFI.git)
cd AD-DIFFI
pip install -r requirements.txt
```
## Usage (Google Colab / Jupyter)

To reproduce the experimental results, follow these steps in your notebook environment:

### 1. Environment Setup
Add the following to your **first cell** to ensure the `src` module is discoverable by the Python interpreter:

```python
import sys
import os

# Set path to the root of the repository
sys.path.append(os.getcwd())
```
### 2. Running an Experiment
You can execute any specific analysis notebook using the `%run` command. For example, to run the **Thyroid subset analysis**:

```python
%run notebooks/07_Real_World_Analysis_Thyroid_subset.ipynb
```
### 3. Handling Kaggle Datasets
For Stroke (05) and Breast Cancer (06) analyses, you will need a Kaggle API token (kaggle.json):
#### 1. When prompted by the notebook, upload your kaggle.json file.
#### 2. The code will automatically configure the Kaggle CLI and download the required data to /tmp/ad_diffi_data/.

## Experiments Summary

This repository contains **8 primary notebooks** to reproduce the results presented in the paper. 

### Theoretical Validation (Notebooks 01-03)
- **`01_Simulation_chapter3_bias_identification.ipynb`**: Identification of **split-frequency bias** in original DIFFI where binary noise is overestimated.
- **`02_Simulation_chapter4_lambda_analysis.ipynb`**: Impact analysis of the **$\lambda$ parameter** on feature scoring.
- **`03_Simulation_chapter5_rso_zscore_validation.ipynb`**: Proof of bias correction using **Root-Split-Only (RSO)** and **Z-score normalization**.

### Practical Validation (Notebooks 04-08)
- **`04_Real_World_Analysis_Annthyroid.ipynb`**: Benchmark using the **full Annthyroid dataset** (ADBench version).
- **`05_Real_World_Analysis_Stroke.ipynb`**: Analysis of the **Kaggle Stroke dataset** with mixed-type features.
- **`06_Real_World_Analysis_Breast_cancer.ipynb`**: Clinical validation including **survival outcome variables** (Status/Duration).
- **`07_Real_World_Analysis_Thyroid_subset.ipynb`**: Detailed feature-level analysis on a **controlled subset (N=1000)** with real column names.
- **`08_Real_World_Analysis_Hepatitis.ipynb`**: Validation on small-sample mixed-type data (**UCI Hepatitis**).

---

## Requirements

The following key libraries are used in this project (Full list in `requirements.txt`):

| Library | Version | Description |
| :--- | :--- | :--- |
| **scikit-learn** | 1.6.1 | Base Isolation Forest implementation |
| **pyod** | 2.0.7 | Anomaly detection benchmarks |
| **pandas** | 2.2.2 | Data manipulation and CSV handling |
| **numpy** | 2.0.2 | Numerical calculations |
| **tabulate** | 0.9.0 | Markdown table formatting |
| **kaggle** | 1.6.0 | Automated dataset acquisition |

---

## Citation

If you use this code or the **AD-DIFFI method** in your research, please cite:

```text
Yu Hidaka, Toru Imai, Katsuhiro Omae. "Adjust DIFFI (AD-DIFFI): 
Robust Feature Importance for Mixed-Type Data in Isolation Forest". (2026).
