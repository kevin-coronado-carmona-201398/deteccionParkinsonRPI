# ================================
# ================================

import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure model and data are loaded
model.eval()

# Move small subset to CPU for SHAP
background = torch.tensor(val_df[feature_cols].sample(200, random_state=42).values, dtype=torch.float)
test_samples = torch.tensor(val_df[feature_cols].sample(500, random_state=0).values, dtype=torch.float)

# Wrapper to ensure SHAP understands forward pass
def model_forward(x):
    with torch.no_grad():
        outputs = model(torch.tensor(x, dtype=torch.float).to(device))
        return torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

# SHAP KernelExplainer (works best for tabular transformers)
explainer = shap.KernelExplainer(model_forward, shap.sample(background.numpy(), 100))
shap_values = explainer.shap_values(test_samples.numpy(), nsamples=100)

# Convert to SHAP array
shap_values = np.array(shap_values)
if shap_values.ndim == 3:  # handle multiclass output
    shap_values = shap_values[1]  # take the 'Patient' class (1)

# ================================
# SHAP Summary Plots
# ================================

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, test_samples.numpy(), feature_names=feature_cols, plot_type='bar', show=False)
plt.title("Feature Importance - SHAP Bar Summary", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, test_samples.numpy(), feature_names=feature_cols, show=False)
plt.title("Feature Impact on Prediction (SHAP Beeswarm Plot)", fontsize=14)
plt.tight_layout()
plt.show()

# ================================
# Optional: Dependence Plot for Top Features
# ================================
top_features = np.argsort(np.abs(shap_values).mean(0))[-3:]  # top 3 important
for i in top_features:
    shap.dependence_plot(i, shap_values, test_samples.numpy(), feature_names=feature_cols)