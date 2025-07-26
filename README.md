# 🏠 California Housing Price Regression - MLOps Workflow

This project implements a complete MLOps pipeline to train a regression model on the California Housing dataset, perform manual quantization, and compare model performance pre- and post-quantization.

---

## 📂 Project Structure

```
A3/
├── src/
│   ├── train.py             # Train and save scikit-learn model
│   ├── quantize.py          # Manual quantization and PyTorch inference
│   ├── comp.py              # Comparison of original and quantized models
│   └── utils.py             # (Optional) Utilities for preprocessing
├── model.joblib             # Trained scikit-learn model
├── test_data.joblib         # Saved test data
├── unquant_params.joblib    # Raw model weights
├── quant_params.joblib      # Quantized model weights
└── README.md                # This file
```

---

## 🧠 Model Description

- **Model**: Linear Regression (Scikit-learn)
- **Dataset**: California Housing
- **Goal**: Predict housing prices
- **Metrics**:
  - `R² Score`: To evaluate regression performance
  - `Model Size`: For storage efficiency

---

## 🔧 Steps Performed

### 1. `train.py`
- Trains a linear regression model using Scikit-learn.
- Splits data into train/test sets and saves the test set using `joblib`.
- Saves the trained model as `model.joblib`.
- Prints R² Score for original model.

### 2. `quantize.py`
- Loads the trained model and extracts `coef_` and `intercept_`.
- Saves unquantized weights in `unquant_params.joblib`.
- Manually quantizes weights using symmetric INT8 quantization.
- Saves quantized weights in `quant_params.joblib`.
- Reconstructs a PyTorch model using dequantized weights.
- Performs inference using PyTorch and computes R² score.

### 3. `comp.py`
- Compares R² score and model sizes.
- Prints final evaluation table.

---

## 📊 Final Results

| Metric         | Original Sklearn Model | Quantized Model     |
|----------------|------------------------|----------------------|
| **R² Score**   | 0.5758                 | 0.5565               |
| **Model Size** | 0.30 KB                | 0.39 KB              |

> ✅ Minimal accuracy loss and successful implementation of quantization.

---

## 📦 Dependencies

```bash
pip install scikit-learn joblib numpy torch
```

---

## ▶️ Usage

### Train the model:
```bash
python src/train.py
```

### Quantize and evaluate:
```bash
python src/quantize.py
```

### Compare metrics:
```bash
python src/comp.py
```

---


