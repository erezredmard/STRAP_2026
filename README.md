# STRAP: Statistical Testing along a Regularization Path

Code for the paper *Regularization-Path Hypothesis Testing of Reconstruction
Errors for Anomaly Detection* (under review).

STRAP trains autoencoders along a path of latent L1-regularization strengths,
tests each test sample's reconstruction error at every strength with a
prediction-interval test on Box-Cox-transformed errors calibrated on a
held-out part of the training data, and combines the per-strength decisions
by voting. The significance level alpha sets the false positive rate.

## Contents

| File | Purpose | Paper |
|---|---|---|
| `strap.py` | Minimal reference implementation of the scoring phase (calibration, p-values, voting) | Sec. 4, Alg. 1 |
| `notebooks/01_adbench_strap.ipynb` | STRAP on the 47 ADBench tabular datasets (clean training with calibration set; also the contaminated protocol) | Sec. 5.1, 6.2 |
| `notebooks/02_adbench_baselines.ipynb` | The 14 ADBench baselines on identical splits | Sec. 5.1, 6.2 |
| `notebooks/03_adbench_baseline_decisions.ipynb` | Label-free thresholds and F1 for the baselines | Sec. 6.2 |
| `notebooks/04_mnistc_strap.ipynb` | STRAP on MNIST-C with a LeNet-type convolutional autoencoder | Sec. 5.2, 6.3 |
| `notebooks/05_medianomaly_strap.ipynb` | STRAP on MedIAnomaly with AE-d* (official benchmark code) | Sec. 5.3, 6.4 |
| `notebooks/06_ablation_grubbs_rosner.ipynb` | Ablation: Grubbs' and Rosner's tests on training errors | App. B |
| `notebooks/07_medianomaly_official_code_check.ipynb` | Reproducibility check with the unmodified official MedIAnomaly code | Sec. 6.4 |

## Running

The notebooks are written for Google Colab with a GPU (we used an NVIDIA A100)
and store their outputs in Google Drive; every notebook resumes after a
disconnect. Run the cells top to bottom.

Data sources (downloaded automatically by the notebooks):
- ADBench: official GitHub repository of ADBench (`adbench/datasets/Classical`).
- MNIST-C: official Zenodo record of MNIST-C.
- MedIAnomaly: official Zenodo record of the MedIAnomaly benchmark; the
  autoencoder and data loaders come from the official MedIAnomaly code.

Each notebook ends with an analysis cell that writes the summary CSV files
used for the tables and figures of the paper.

## Quick check of the decision layer

```bash
python strap.py
```

prints the per-strength false positive rate on simulated normal data, which
should be close to alpha = 0.05.
