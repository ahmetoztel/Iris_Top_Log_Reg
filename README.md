# Iris_Top_Log_Reg
Logistic regression-based iris recognition using Betti numbers and topological feature vectors derived from 2D image analysis.
# Iris Topological Features - Logistic Regression Classifier

This repository contains a complete implementation of a biometric identification experiment using topological features extracted from 2D iris images. The classification task is performed using **Logistic Regression** after applying proper normalization and PCA-based dimensionality reduction.

The topological features include:
- Betti₀ (number of connected components),
- Betti₁ (number of holes),
- Betti₁/Betti₀ ratio

These features were previously computed and saved in `image_results.xlsx`.

---

## Overview

The classification pipeline follows these steps:

1. **Load topological features** from Excel
2. **Extract person identity** from file names
3. **Normalize** the feature vectors using `MinMaxScaler` (fit only on training data)
4. **Reduce dimensionality** via PCA (`n_components=0.99`, fit only on training data)
5. **Train Logistic Regression** classifier (`C=10`, `max_iter=2000`)
6. **Repeat the process 10 times** with different random seeds
7. **Evaluate accuracy** and save:
   - Per-run accuracy scores
   - Accuracy distribution boxplot
   - Summary statistics (mean ± std)

---

## Files

- `logreg_results_corrected.xlsx`: Accuracy values and summary statistics
- `logreg_boxplot_corrected.png`: Boxplot of accuracy across 10 repeats
- `logistic_regression_topological.py`: Main source code

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl

Install all dependencies with:

```bash
pip install -r requirements.txt
