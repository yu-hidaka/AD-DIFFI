# AD-DIFFI: Robust Feature Importance for Mixed-Type Data in Isolation Forest

Official implementation of the paper: 
**"Adjust DIFFI (AD-DIFFI): Robust Feature Importance for Mixed-Type Data in Isolation Forest"**

## Overview
AD-DIFFI (Adjusted DIFFI) is a feature importance method for Isolation Forest, specifically optimized for mixed-type datasets containing both continuous and binary features. It addresses two primary biases found in the original DIFFI (Depth-based Isolation Forest Feature Importance) method:

1.  **Overestimation of Binary Noise**: Prevents binary features from being incorrectly ranked as important due to high split frequency in deep nodes.
2.  **Underestimation of Binary Signals**: Ensures that binary features that isolate anomalies early (near the root) are correctly valued.

The core components of the method are **Root-Split-Only (RSO)** constraints and **Noise-based Z-normalization**.

## Repository Structure
- **src/**: Contains the core implementation of the AD-DIFFI algorithm.
- **notebooks/**: Contains Jupyter Notebooks for reproducing the results presented in the paper.
    - `01_Simulations.ipynb`: Validates the bias correction performance using synthetic data.
    - `02_Real_world_data_analysis.ipynb`: Evaluates performance on clinical benchmark datasets (Annthyroid, Breast Cancer, and Hepatitis).
- **requirements.txt**: List of Python dependencies.

## Installation
This project requires Python 3.12 or later. To set up the environment, clone the repository and install the dependencies:

```bash
git clone [https://github.com/yu-hidaka/AD-DIFFI.git](https://github.com/yu-hidaka/AD-DIFFI.git)
cd AD-DIFFI
pip install -r requirements.txt
