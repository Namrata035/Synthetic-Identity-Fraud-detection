import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Example: Replace these with your real model predictions and true labels
# y_true = actual class labels (0 = Not Fraud, 1 = Fraud)
# y_pred_cnn, y_pred_lstm, y_pred_ensemble = predicted probabilities for class 1
# Example simulated data for structure
import numpy as np
np.random.seed(42)
y_true = np.random.randint(0, 2, 500)
y_pred_cnn = np.random.rand(500)
y_pred_lstm = np.random.rand(500)
y_pred_ensemble = np.random.rand(500)

# Compute Precision–Recall curves
precision_cnn, recall_cnn, _ = precision_recall_curve(y_true, y_pred_cnn)
precision_lstm, recall_lstm, _ = precision_recall_curve(y_true, y_pred_lstm)
precision_ens, recall_ens, _ = precision_recall_curve(y_true, y_pred_ensemble)

# Compute AUC for each model
auc_cnn = auc(recall_cnn, precision_cnn)
auc_lstm = auc(recall_lstm, precision_lstm)
auc_ens = auc(recall_ens, precision_ens)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(recall_cnn, precision_cnn, label=f'CNN (AUC={auc_cnn:.2f})')
plt.plot(recall_lstm, precision_lstm, label=f'LSTM (AUC={auc_lstm:.2f})')
plt.plot(recall_ens, precision_ens, label=f'Ensemble (AUC={auc_ens:.2f})')

plt.title('Graph 2: Precision–Recall Curve for CNN, LSTM, and Ensemble', fontsize=13, fontweight='bold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
