import joblib
import os
from sklearn.metrics import r2_score

# Load test data
X_test, y_test = joblib.load("test_data.joblib")

# Load original sklearn model
model_sklearn = joblib.load("model.joblib")
y_pred_sklearn = model_sklearn.predict(X_test)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

# Load quantized parameters and reconstruct PyTorch model
import torch
import torch.nn as nn
import numpy as np

quant_params = joblib.load("quant_params.joblib")
quantized_coef = quant_params["quantized_coef"]
quantized_intercept = quant_params["quantized_intercept"]
scale = quant_params["scale"]

# Dequantize
dq_coef = quantized_coef.astype(np.float32) * scale
dq_intercept = quantized_intercept * scale

class QuantLinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

model_pt = QuantLinearModel(X_test.shape[1])
with torch.no_grad():
    model_pt.fc.weight.copy_(torch.tensor(dq_coef[np.newaxis, :], dtype=torch.float32))
    model_pt.fc.bias.copy_(torch.tensor([dq_intercept], dtype=torch.float32))

# Inference
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_quant = model_pt(X_tensor).numpy()

r2_quant = r2_score(y_test, y_pred_quant)

# Model sizes
unquant_size = os.path.getsize("unquant_params.joblib") / 1024
quant_size = os.path.getsize("quant_params.joblib") / 1024

# Print table
print(f"{'Metric':<20} {'Original Sklearn Model':<30} {'Quantized Model'}")
print("-" * 80)
print(f"{'R² Score':<20} {r2_sklearn:<30.4f} {r2_quant:.4f}")
print(f"{'Model Size (KB)':<20} {unquant_size:<30.2f} {quant_size:.2f}")
