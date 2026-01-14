# Skeleton-based-Violence-Detection

## 📌 Project Overview

This project implements a pose-based violence detection system using human skeleton keypoints extracted from videos and a Bidirectional LSTM (BiLSTM) neural network for temporal classification.

Instead of raw RGB video frames, the model operates on human pose sequences, making it computationally efficient and privacy-preserving.

## 🗂️ Project Folder Structure (Original Design)

```
Skeleton-based-Violence-Detection/
│
├── dataset/ 
│   └── RWF2000/ 
│       ├── train/
│       │   ├── Fight/
│       │   └── NonFight/
│       └── val/
│           ├── Fight/
│           └── NonFight/
│
├── extracted_keypoints/ 
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
│
├── src/
│   ├── pose_extractor.py
│   ├── dataset_loader.py
│   ├── model.py
│   └── predict.py
│
├── scripts/
│   ├── extract_keypoints.py
│   ├── train_bilstm.py
│   ├── infer_video.py
│   ├── evaluate_model.py
│   └── plot_metrics.py
│
├── models/
│   └── bilstm_model_165.h5
│
├── requirements.txt
└── README.md
```
---
## 🎯 Objective

To classify videos into:

- Fight (Violence)
- NonFight (Non-Violence)

using temporal pose information extracted from video frames.

---
## 🔧 Technologies Used

- Python 3.12
- MediaPipe Pose
- OpenCV
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---
## 🔍 Methodology
### 1️⃣ Pose Extraction

- Each video is processed frame-by-frame
- MediaPipe Pose extracts 33 body landmarks
- Each landmark contributes: ``` x, y, visibility```
- Total features per frame = ``` 33 × 3 = 99```
- Saved as: ```(frames, 99) → .npy files```

### 2️⃣ Sequence Normalization

- Variable-length sequences are:
- Padded
- Truncated
- Final input shape:
```
(samples, 30 frames, 99 features)
```

### 3️⃣ Model Architecture

#### Bidirectional LSTM (BiLSTM)
```
Input (30 × 99)
 → Masking
 → BiLSTM (128 units)
 → Dropout
 → BiLSTM (64 units)
 → Dropout
 → Dense (ReLU)
 → Dense (Sigmoid)
```

Loss: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

📈 Observations

Accuracy: 0.6268221574344023

 Classification Report:
 ```
               precision    recall  f1-score   support

     NonFight       0.64      0.42      0.51       157
        Fight       0.62      0.80      0.70       186

     accuracy                           0.63       343
    macro avg       0.63      0.61      0.60       343
 weighted avg       0.63      0.63      0.61       343
```

Confusion Matrix: 

<img width="402" height="387" alt="image" src="https://github.com/user-attachments/assets/328b29f3-d9bb-4913-a1b5-0675812e7e8d" />

ROC Curve:

<img width="402" height="387" alt="image" src="https://github.com/user-attachments/assets/8583972a-9eed-4267-9b9d-69e23e9e0112" />

Prediction Confidence Distribution:

<img width="402" height="387" alt="image" src="https://github.com/user-attachments/assets/b28344de-8f9e-4a97-83a0-7fc1cb375d16" />

## ▶️ How to Run the Project
### 1️⃣ Clone the Repository
```
git clone https://github.com/Madhu-Wala/Skeleton-based-Violence-Detection.git
cd Skeleton-based-Violence-Detection
```

### 2️⃣ Create & Activate Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

#### ⚠️ Important Notes

TensorFlow runs on CPU (Windows limitation)

Dataset is not included (size ≈ 12GB)

### 4️⃣ Download Dataset Manually

Download RWF-2000 dataset from Kaggle:

🔗 [https://www.kaggle.com/datasets/vulamnguyen/rwf2000](https://www.kaggle.com/datasets/vulamnguyen/rwf2000)

Extract and place it as:
```
dataset/
└── RWF2000/
    ├── train/
    │   ├── Fight/
    │   └── NonFight/
    └── val/
        ├── Fight/
        └── NonFight/
```

### 5️⃣ Extract Pose Keypoints

⚠️ This is a heavy step and may take 3-4 hours on CPU.
```
python scripts/extract_keypoints.py
```

This generates:
```
extracted_keypoints/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

Each .npy file contains:
```
(frames, 99)
```

### 6️⃣ Train BiLSTM Model
```
python scripts/train_bilstm.py
```

Outputs:
```
models/bilstm_model_165.h5
```

Training time: ~10–20 minutes (CPU)

Accuracy ~55–60%

### 7️⃣ Plot Performance Metrics
```
python scripts/plot_metrics.py
```

### 8️⃣ Run Inference on a Video

Edit video path inside:
```
scripts/infer_video.py
```

Then run:
```
python scripts/infer_video.py
```

Output:
```
Prediction: Fight / NonFight
Confidence: 0.xx
```

## ⚠️ Common Issues
### ❌ Low Accuracy (~0.55)

- ✔ Expected for pose-only approach
- ✔ Dataset contains ambiguous actions

### ❌ Some videos skipped during extraction

✔ MediaPipe fails on:
- Corrupt videos
- No detectable human pose
- Occlusions

