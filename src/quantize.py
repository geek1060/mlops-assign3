from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time
import sys
from joblib import dump, load
import os
import torch
import torch.nn as nn
import pdb
import pickle

# Helper Functions
def quantize_input(inp, n_bits=3):
    levels = 2 ** n_bits -1
    val =  np.round(inp * levels).astype(np.uint8)
    return val

def dequantize_model(inp, n_bits=3):
    levels = 2 ** n_bits -1
    val =  (inp.astype(np.float64) / levels)
    return val

# Load data
digits = load_digits()
X, y = digits.data, digits.target

print(f"Max value:{X.max()}, Min value:{X.min()}")

# 1. Unquantized model on unnormalized data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
dump(model, "unquantized_model_unquantized_unnormalized_data.joblib")
print(f"Unquantized Model, Unquantized Data, Acc={acc*100:.4f}, Size={os.path.getsize('unquantized_model_unquantized_unnormalized_data.joblib')/1024:.2f} KB")
print("===============================================================================")

# 2. Unquantized model on normalized data
X_norm = (X - X.min()) / (X.max() - X.min())
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
dump(model, "unquantized_model_unquantized_normalized_data.joblib")
print(f"Unquantized Model, Normalized Data, Acc={acc:.4f}, Size={os.path.getsize('unquantized_model_unquantized_normalized_data.joblib')/1024:.2f} KB")
print("===============================================================================")

# 3. Unquantized model on quantized normalized data (float)
X = digits.data / 16.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_q = quantize_input(X_train, 3)
X_test_q = quantize_input(X_test, 3)
model.fit(X_train_q, y_train)
y_pred = model.predict(X_test_q)
acc = accuracy_score(y_test, y_pred)
dump(model, "unquantized_model_normalized_quantized_data_int.joblib")
print(f"Unquantized Model, Normalized Quantized Data Int, Acc={acc:.4f}, Size={os.path.getsize('unquantized_model_normalized_quantized_data_int.joblib')/1024:.2f} KB")
print("===============================================================================")

# 4. Quantized weights, normalized int data
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_q, y_train)
weights = np.asarray(model.coef_)
bias = np.asarray(model.intercept_)

w_min, w_max = weights.min(), weights.max()
b_min, b_max = bias.min(), bias.max()
weights_norm = (weights - w_min)/(w_max - w_min)
bias_norm = (bias - b_min)/(b_max - b_min)
quantized_weights = quantize_input(weights_norm, 4)
quantized_bias = quantize_input(bias_norm, 4)

dump({"weights": weights, "bias": bias}, "unquantized_model_weights_quantized_normalized_int_data.joblib")
dump({"weights": quantized_weights, "bias": quantized_bias}, "quantized_model_weights_quantized_normalized_int_data.joblib")

weights_rec = dequantize_model(quantized_weights, 4) * (w_max - w_min) + w_min
bias_rec = dequantize_model(quantized_bias, 4) * (b_max - b_min) + b_min
outputs = np.dot(X_test_q, weights_rec.T) + bias_rec
y_pred = np.argmax(outputs, axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"Quantized Weights, Normalized Quantized Int Data, Acc={acc:.4f}, Quantized Size={os.path.getsize('quantized_model_weights_quantized_normalized_int_data.joblib')/1024:.2f} KB")
print("===============================================================================")

# 5. PyTorch quantized model
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_q, y_train)
weights = np.asarray(model.coef_, dtype=np.float32)
bias = np.asarray(model.intercept_, dtype=np.float32)

class LogisticRegressionTorch(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.linear(x)

pt_model = LogisticRegressionTorch(weights.shape[1], weights.shape[0])
pt_model.linear.weight.data = torch.from_numpy(weights)
pt_model.linear.bias.data = torch.from_numpy(bias)
quantized_model = torch.quantization.quantize_dynamic(pt_model, {nn.Linear}, dtype=torch.qint8)

X_input = torch.from_numpy(X_test_q.astype(np.float32))
y_pred = torch.argmax(quantized_model(X_input), axis=1).numpy()
acc = accuracy_score(y_test, y_pred)

dump(pt_model, "torch_unquantized_model_quantized_normalized_int_data.joblib")
dump(quantized_model, "torch_quantized_model_quantized_normalized_int_data.joblib")
print(f"Torch Quantized Model, Acc={acc:.4f}, Size={os.path.getsize('torch_quantized_model_quantized_normalized_int_data.joblib')/1024:.2f} KB")
print("===============================================================================")
