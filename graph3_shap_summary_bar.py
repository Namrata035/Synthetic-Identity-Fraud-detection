import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example: load your dataset and trained model
# model = your_trained_model
# X_test = your feature matrix for testing

# For demonstration (synthetic example):
feature_names = ['Credit_Score', 'Transaction_Amount', 'Account_Age', 
                 'Num_Transactions', 'Time_Since_Last_Activity', 
                 'Loan_Amount', 'Employment_Length', 'Income', 
                 'Debt_Ratio', 'Open_Credit_Lines', 'Utilization_Rate', 'Delinquencies']

X_test = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
explainer = shap.Explainer(lambda x: np.random.rand(x.shape[0], 2), X_test)  # Dummy model output
shap_values = explainer(X_test)

# SHAP bar chart for top 10 features
shap.plots.bar(shap_values[:, :, 1], max_display=10, show=False)
plt.title("Graph 3: SHAP Summary Bar Chart (Top 10 Features)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
