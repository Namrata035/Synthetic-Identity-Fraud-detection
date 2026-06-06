import matplotlib.pyplot as plt
import numpy as np

# Example: Replace these with your actual model accuracies
epochs = np.arange(1, 21)   # 1 to 20 epochs
cnn_acc = [0.60, 0.63, 0.66, 0.68, 0.70, 0.72, 0.73, 0.74, 0.75, 0.76,
           0.77, 0.78, 0.79, 0.80, 0.81, 0.81, 0.82, 0.83, 0.83, 0.84]
lstm_acc = [0.58, 0.61, 0.64, 0.68, 0.71, 0.74, 0.77, 0.79, 0.81, 0.83,
            0.84, 0.85, 0.86, 0.87, 0.88, 0.88, 0.89, 0.90, 0.90, 0.91]
ensemble_acc = [0.62, 0.66, 0.70, 0.73, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85,
                0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.91, 0.92, 0.92, 0.93]

# Plot the accuracies
plt.figure(figsize=(8, 5))
plt.plot(epochs, cnn_acc, marker='o', label='CNN')
plt.plot(epochs, lstm_acc, marker='s', label='LSTM')
plt.plot(epochs, ensemble_acc, marker='^', label='Ensemble (CNN+LSTM)')

# Graph details
plt.title('Graph 1: Accuracy vs Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()
