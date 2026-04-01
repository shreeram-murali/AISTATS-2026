# Reproducibility Guide for Paper Submission

This repository contains code and instructions to reproduce the main results of our paper:

**A Computationally Efficient Classifier with Frequentist Bounds on Prediction Errors**

## Directory Structure

- `ecg.ipynb` — Experiments on ECG heartbeat classification.
- `margin.ipynb` — Experiments on synthetically genearted data separated by a margin.
- `nwc_mnist.ipynb` — Nadaraya-Watson classifier experiments on MNIST.
- `synthetic.ipynb` — Experiments on synthetic Lipschitz-continuous datasets.
- `hyperparameter-optimization/` — Scripts and notebooks for hyperparameter tuning.
- `reml/` — Core implementation of the classifier and supporting modules.
- `setup.py` — Install script for dependencies.

## Setup Instructions

1. **Install dependencies**
	It is recommended to use a Python 3.8+ environment. Install required packages:
	```bash
	pip install -e .
	```

2. **Data Preparation**
	- MNIST and other datasets are expected in the `data/` folders. If not present, download them as instructed in the relevant notebook cells. The ECG dataset can be found here: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

## Running Experiments

- **ECG Classification**: Open and run `ecg.ipynb`.
- **MNIST Classification**: Open and run `nwc_mnist.ipynb`.
- **Synthetic Data**: Open and run `synthetic.ipynb`.
- **Margin Analysis**: Open and run `margin.ipynb`.
- **Hyperparameter Optimization**: See `hyperparameter-optimization/` for scripts and results. Run `main.py` for optimization.

All notebooks are self-contained. 

## Notes

- For any questions or issues, please feel free to mention them in the reviews.

