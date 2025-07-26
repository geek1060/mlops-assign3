import joblib
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import r2_score

# Load trained model and test data
model = joblib.load("model.joblib")
X_test, y_test = joblib.load("test_data.joblib")

# Extract weights
coef = model.coef_.astype(np.float32)
intercept = float(model.intercept_)

# Save unquantized parameters
unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")

# Manual quantization (symmetric int8 with zero-point = 0)
scale = np.max(np.abs(coef)) / 127.0
quantized_coef = np.clip(np.round(coef / scale), -128, 127).astype(np.int8)
quantized_intercept = int(np.round(intercept / scale))

# Save quantized parameters
quant_params = {
    "quantized_coef": quantized_coef,
    "quantized_intercept": quantized_intercept,
    "scale": scale
}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize
dq_coef = quantized_coef.astype(np.float32) * scale
dq_intercept = quantized_intercept * scale

# Define PyTorch model
class QuantLinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

# Initialize model
model_pt = QuantLinearModel(X_test.shape[1])
with torch.no_grad():
    model_pt.fc.weight.copy_(torch.tensor(dq_coef[np.newaxis, :], dtype=torch.float32))
    model_pt.fc.bias.copy_(torch.tensor([dq_intercept], dtype=torch.float32))

# Inference
X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_test, dtype=torch.float32)

with torch.no_grad():
    y_pred_pt = model_pt(X_tensor)

# Evaluation
r2_quant = r2_score(y_test, y_pred_pt.numpy())
r2_original = r2_score(y_test, model.predict(X_test))

# File sizes
unquant_size = os.path.getsize("unquant_params.joblib") / 1024
quant_size = os.path.getsize("quant_params.joblib") / 1024

# Final output
print("\nMetric                      | Original Sklearn Model | Quantized PyTorch Model")
print("---------------------------|-------------------------|---------------------------")
print(f"R² Score                   | {r2_original:.4f}                  | {r2_quant:.4f}")
print(f"Model Size (KB)           | {unquant_size:.2f} KB               | {quant_size:.2f} KB")
