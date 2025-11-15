# 🪖 Helmet Detection: Traditional ML vs Deep Learning with LoRA

Comparative study of helmet detection using Traditional Machine Learning (Feature Engineering) and Deep Learning (Transfer Learning with LoRA fine-tuning).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project compares two approaches for binary helmet detection classification:

- **No Helmet** (Class 0)
- **With Helmet** (Class 1)

### Approaches Implemented:

**1. Traditional Machine Learning (Feature Engineering)**

- **SET 1:** Histogram + HOG + LBP
- **SET 2:** Color Moments + HOG + GLCM
- **SET 3:** Edge Features + HOG + Color Histogram
- **Algorithms:** SVM (RBF & Linear), Random Forest, Gradient Boosting

**2. Deep Learning (Transfer Learning + LoRA)**

- **Architecture:** MobileNetV2 (ImageNet pre-trained)
- **Strategy:** LoRA (Low-Rank Adaptation) fine-tuning
- **Trainable Parameters:** Only 128-dimensional adapter layer (~0.5% of total)

---

## 📊 Dataset

### Source: Roboflow Universe

**Dataset:** [h3Lm 40L0 Oid XD Model](https://universe.roboflow.com/aa-cqfub/h3lm-40l0-oid-xd-3dofr)

**Statistics:**

- **Total Images:** 1,334 cropped bounding boxes
- **Classes:** 2 (no_helmet: 837, with_helmet: 502)
- **Format:** YOLO v11
- **Version:** 5
- **Image Size:** 128x128 (resized)

### Quick Access Options:

#### Option A: Download from Roboflow (Recommended)

```python
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # Get your API key from Roboflow
project = rf.workspace("aa-cqfub").project("h3lm-40l0-oid-xd-3dofr")
version = project.version(5)
dataset = version.download("yolov11")
```

**Get your free API key:** [Roboflow Sign Up](https://app.roboflow.com/)

#### Option B: Use Local Dataset

Clone this repository and use the dataset in the `dataset/` folder.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Dataset Structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── README.dataset.txt
```

---

## 🔬 Methods

### Part 1: Traditional Machine Learning

**Preprocessing Pipeline:**

- Crop bounding boxes from full images
- Resize to 128×128
- Data augmentation (Gaussian blur, flip, rotation, brightness/contrast)

**Feature Extraction (3 Sets):**

| Set   | Features                                     | Dimensions |
| ----- | -------------------------------------------- | ---------- |
| SET 1 | Histogram (96) + HOG (~8100) + LBP (26)      | ~8222      |
| SET 2 | Color Moments (9) + HOG (~576) + GLCM (20)   | ~605       |
| SET 3 | Edge (5) + HOG (~432) + Color Histogram (48) | ~485       |

**Post-processing:**

- StandardScaler normalization
- PCA (95% variance retention)
- SMOTE for class imbalance

**Algorithms Tested:**

- SVM with RBF kernel
- SVM with Linear kernel
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)

### Part 2: Deep Learning with LoRA

**Architecture:**

- Base: MobileNetV2 (ImageNet pre-trained, frozen)
- Adapter: Dense(128, ReLU) → Dense(2, Softmax)
- Total params: ~2.4M (trainable: ~164K, 6.8%)

**Training Configuration:**

- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Max epochs: 50
- Early stopping: patience=10
- Learning rate reduction: factor=0.5, patience=5

**Data Split:**

- Train: 80% (with augmentation)
- Validation: 10%
- Test: 10%

---

## 🚀 Installation

### Prerequisites:

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- Google Colab (recommended) or local Jupyter

### Install Dependencies:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install required packages
pip install -r requirements.txt
```

**Or manually install:**

```bash
pip install roboflow albumentations opencv-python scikit-learn scikit-image
pip install matplotlib seaborn imbalanced-learn tensorflow>=2.13.0
pip install pandas numpy jupyter
```

---

## 💻 Usage

### Option 1: Google Colab (Recommended)

1. Upload notebook to Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run all cells sequentially

### Option 2: Local Jupyter

```bash
# Start Jupyter
jupyter notebook

# Open notebook
Machine_Learning_088_095_rs_part_1&2.ipynb
```

## 📈 Results

### Quantitative Results:

| Method                     | Algorithm          | Test Accuracy | Precision  | Recall     |
| -------------------------- | ------------------ | ------------- | ---------- | ---------- |
| **Traditional ML - SET 1** | SVM-RBF            | 88.50%        | 0.8842     | 0.8850     |
| **Traditional ML - SET 2** | SVM-RBF            | 87.30%        | 0.8725     | 0.8730     |
| **Traditional ML - SET 3** | SVM-RBF            | 86.80%        | 0.8673     | 0.8680     |
| **Deep Learning**          | MobileNetV2 + LoRA | **89.52%**    | **0.8955** | **0.8952** |

**Improvement:** Deep Learning achieves +1.02% accuracy over best Traditional ML method.

---

## 📦 Requirements

### Python Packages:

```
roboflow>=1.0.0
albumentations>=1.3.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
scikit-image>=0.20.0
matplotlib>=3.7.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0
tensorflow>=2.13.0
pandas>=1.5.0
numpy>=1.24.0
jupyter>=1.0.0
```

**Full requirements:** See `requirements.txt`
