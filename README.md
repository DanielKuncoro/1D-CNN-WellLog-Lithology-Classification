# 1D-CNN Well Log Lithology Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)

A 1D Convolutional Neural Network (1D-CNN) for automated lithology classification from well log data, evaluated using a **Leave-One-Well-Out (LOWO)** cross-validation scheme. This project was developed as part of an undergraduate thesis focusing on coal seam identification in subsurface formations.

---

## Overview

Manual lithology interpretation from well logs is time-consuming and subject to interpreter bias. This project applies deep learning to automate the classification of four lithological classes — **Claystone, Sandstone, Coaly Shale, and Coal** — using Gamma Ray (GR) and Density (RHOB) logs as input features.

The LOWO scheme ensures that each well is tested as a completely blind well (no data leakage between training and test sets), providing a realistic estimate of model generalization performance across different wells.

---

## Lithology Classes

| Code | Lithology    |
|------|--------------|
| 0    | Claystone    |
| 1    | Sandstone    |
| 2    | Coaly Shale  |
| 3    | Coal         |

---

## Methodology

### Input Features
- **GR** — Gamma Ray log
- **DENSITY (RHOB)** — Bulk density log

### Sliding Window
Each sample is constructed by extracting a centered window of 101 depth samples around the target depth point. This provides local stratigraphic context for the classifier.

### CNN Architecture

```
Input: (101, 2)
  ↓
Conv1D(64, kernel=7, ELU) → BatchNorm → MaxPool(2) → Dropout(0.2)
  ↓
Conv1D(128, kernel=5, ELU) → BatchNorm → MaxPool(2) → Dropout(0.2)
  ↓
Conv1D(256, kernel=3, ELU) → BatchNorm → Dropout(0.2)
  ↓
Flatten → Dense(128, ELU, L2) → Dropout(0.3)
  ↓
Dense(4, Softmax)
```

### Training Strategy
| Component         | Setting                                      |
|-------------------|----------------------------------------------|
| Optimizer         | Adam (lr = 1e-3)                             |
| Loss Function     | Focal Loss (γ = 2.0)                         |
| Class Weighting   | Balanced, capped at 6.0                      |
| Validation Split  | 10% stratified                               |
| Early Stopping    | Patience = 7 (monitor: val_loss)             |
| LR Scheduler      | ReduceLROnPlateau (factor=0.5, patience=4)   |
| Batch Size        | 64                                           |
| Max Epochs        | 100                                          |

Focal Loss was chosen over standard cross-entropy to handle class imbalance, particularly for the minority class (Coaly Shale). Class weights are additionally applied with a cap to prevent extreme over-weighting.

---

## Repository Structure

```
1D-CNN-WellLog-Lithology-Classification/
│
├── lowo_cnn_lithology.py       # Main training and evaluation script
│
├── data/
│   ├── Well_A.csv
│   ├── Well_B.csv
│   └── ...                     # 9 wells total (CSV format)
│
├── outputs/
│   ├── cm_Well_A.png           # Confusion matrix per well
│   ├── history_Well_A.png      # Training history per well
│   ├── pred_Well_A.csv         # Prediction output per well
│   ├── lowo_summary.csv        # Accuracy summary across all folds
│   └── lowo_accuracy_summary.png
│
├── requirements.txt
└── README.md
```

---

## Input Data Format

Each well must be provided as a `.csv` file with at least the following columns:

| Column    | Description              |
|-----------|--------------------------|
| `DEPTH`   | Measured depth (m)       |
| `GR`      | Gamma Ray (API)          |
| `DENSITY` | Bulk density (g/cc)      |
| `LITHO`   | Lithology label (0–3)    |

Rows with missing values in any of these columns are automatically dropped.

---

## Results

The model was evaluated across **9 wells** using Leave-One-Well-Out (LOWO) cross-validation. Each fold trains on 8 wells and tests on the remaining 1 blind well.

### LOWO Accuracy per Blind Well

| Well     | Accuracy |
|----------|----------|
| Well_1   | —        |
| Well_2   | —        |
| ...      | ...      |
| **Mean** | **—**    |
| **Std**  | **—**    |

> *Replace the table above with your actual results from `outputs/lowo_summary.csv`.*

### Example: Confusion Matrix

> *Insert `outputs/cm_<WellName>.png` here.*

### Example: Training History

> *Insert `outputs/history_<WellName>.png` here.*

---

## Getting Started

### Requirements

```
tensorflow>=2.10
scikit-learn
numpy
pandas
matplotlib
seaborn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Script

1. Place all well CSV files in your `DATA_DIR` folder.
2. Update `DATA_DIR` and `OUTPUT_DIR` paths in `lowo_cnn_lithology.py`.
3. Run:

```bash
python lowo_cnn_lithology.py
```

Output files (confusion matrices, training curves, prediction CSVs, and summary) will be saved to `OUTPUT_DIR`.

---

## Configuration

Key parameters can be adjusted at the top of `lowo_cnn_lithology.py`:

```python
WINDOW_SIZE      = 101     # Sliding window length (depth samples)
FEATURES         = ["GR", "DENSITY"]
N_CLASSES        = 4
EPOCHS           = 100
BATCH_SIZE       = 64
MAX_CLASS_WEIGHT = 6.0     # Cap on class weight to prevent extreme imbalance
```

---

## Citation

If you use this code in your research or academic work, please cite:

```
Kuncoro, D. (2025). 1D-CNN Well Log Lithology Classification using
Leave-One-Well-Out Cross-Validation. Undergraduate Thesis.
```

---

## License

This project is licensed under the MIT License.
