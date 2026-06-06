# graph4_model_comparison.py

import matplotlib.pyplot as plt
import numpy as np

# 🔹 Example accuracies (you can replace these with your actual model results)
models = ['Logistic Regression', 'Random Forest', 'CNN', 'LSTM', 'CNN–LSTM (Proposed)']
accuracies = [0.82, 0.88, 0.63, 0.92, 0.94]
f1_scores = [0.70, 0.83, 0.61, 0.89, 0.93]

# Set up the bar positions
x = np.arange(len(models))
width = 0.35

# 🔹 Create the bar chart
plt.figure(figsize=(10,6))
bars1 = plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#4C72B0')
bars2 = plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#55A868')

# Add labels and title
plt.xlabel("Models", fontsize=12)
plt.ylabel("Performance", fontsize=12)
plt.title("Comparison with Existing Models", fontsize=14, fontweight='bold')
plt.xticks(x, models, rotation=20, ha='right')
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bar values
for bar in bars1 + bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.2f}', ha='center', fontsize=9)

# Save and show
plt.tight_layout()
plt.savefig("Graph4_Model_Comparison.png", dpi=300)
plt.show()

print("✅ Graph saved as: Model_Comparison.png")
