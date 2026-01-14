import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from keras.models import load_model

from src.dataset_loader import load_dataset

# Load validation data
X_val, y_val = load_dataset("extracted_keypoints/val")

# Load trained model
model = load_model("models/bilstm_model_165.h5")

# Predict
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["NonFight", "Fight"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

#output:
# Accuracy: 0.6268221574344023

# Classification Report:
#               precision    recall  f1-score   support

#     NonFight       0.64      0.42      0.51       157
#        Fight       0.62      0.80      0.70       186

#     accuracy                           0.63       343
#    macro avg       0.63      0.61      0.60       343
# weighted avg       0.63      0.63      0.61       343


# Confusion Matrix:
# [[ 66  91]
#  [ 37 149]]
