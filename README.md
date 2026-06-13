# 🌸 Decode Labs Project 2 — Iris KNN Classifier

A supervised machine learning project that classifies Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm with scikit-learn.

---

## 📌 Project Goals

| Requirement | Status |
|---|---|
| Load & explore the Iris dataset | ✅ |
| Train / Test split (80/20) | ✅ |
| Normalization (MinMaxScaler) | ✅ |
| Standardization (StandardScaler) | ✅ |
| KNN classification algorithm | ✅ |
| Confusion Matrix | ✅ |
| F1-Score (macro, weighted, per-class) | ✅ |
| Full Classification Report | ✅ |
| Visual Dashboard | ✅ |

---

## 🗂️ Project Structure

```
Decode Labs Project 2/
│
├── iris_classifier.py   # Main ML pipeline script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the classifier

```bash
python iris_classifier.py
```

---

## 🔄 Pipeline Overview

```
Raw Iris Data (150 samples, 4 features, 3 classes)
        │
        ▼
  Train / Test Split  ──── 80% train │ 20% test (stratified)
        │
        ▼
  Preprocessing
    ├─ Normalization   → MinMaxScaler  [0, 1]
    └─ Standardization → StandardScaler  (mean=0, std=1)
        │
        ▼
  Optimal K Selection ──── 5-Fold Cross-Validation (K = 1..20)
        │
        ▼
  KNN Classifier (Euclidean distance)
        │
        ▼
  Evaluation
    ├─ Accuracy
    ├─ Confusion Matrix
    ├─ F1-Score (macro / weighted / per-class)
    └─ Full Classification Report
        │
        ▼
  Visual Dashboard  →  iris_knn_results.png
```

---

## 📊 Key Concepts

### Normalization (MinMaxScaler)
Scales each feature to the range **[0, 1]**:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

### Standardization (StandardScaler)
Centers features to **mean = 0, std = 1** — preferred for distance-based algorithms like KNN:

```
X_std = (X - mean) / std
```

### K-Nearest Neighbors (KNN)
A non-parametric algorithm that classifies a sample by majority vote of its **K nearest neighbors** in feature space using Euclidean distance.

### Metrics
| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions / total |
| Precision | TP / (TP + FP) per class |
| Recall | TP / (TP + FN) per class |
| F1-Score | Harmonic mean of Precision & Recall |
| Confusion Matrix | Actual vs. Predicted class breakdown |

---

## 🌿 Dataset Info

The **Iris dataset** contains 150 samples of 3 species:

| Species | Count |
|---|---|
| Iris-Setosa | 50 |
| Iris-Versicolor | 50 |
| Iris-Virginica | 50 |

**Features**: Sepal Length, Sepal Width, Petal Length, Petal Width (all in cm)

---

*Built as part of the Decode Labs Internship Program.*
