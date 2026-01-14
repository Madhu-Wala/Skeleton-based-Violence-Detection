import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from keras.models import load_model

from src.dataset_loader import load_dataset

# ----------------------------
# Load data & model
# ----------------------------
X_val, y_val = load_dataset("extracted_keypoints/val")
model = load_model("models/bilstm_model_165.h5")

# Predict
y_prob = model.predict(X_val).ravel()
y_pred = (y_prob > 0.5).astype(int)

# # ----------------------------
# # Print metrics 
# # ----------------------------
# print("\nClassification Report:")
# print(classification_report(y_val, y_pred, target_names=["NonFight", "Fight"]))

# ----------------------------
# 1️⃣ Confusion Matrix Plot
# ----------------------------
cm = confusion_matrix(y_val, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["NonFight", "Fight"])
plt.yticks([0, 1], ["NonFight", "Fight"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ----------------------------
# 2️⃣ ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ Prediction Confidence Distribution
# ----------------------------
plt.figure()
plt.hist(y_prob[y_val == 0], bins=30, alpha=0.6, label="NonFight")
plt.hist(y_prob[y_val == 1], bins=30, alpha=0.6, label="Fight")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Prediction Confidence Distribution")
plt.legend()
plt.tight_layout()
plt.show()
