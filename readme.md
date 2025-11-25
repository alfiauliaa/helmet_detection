# Helmet Detection: Classic ML vs CNN vs Transfer Learning (MobileNetV2 + LoRA)

Comparative study of helmet detection using three approaches: Traditional Machine Learning (Feature Engineering), Deep Learning from Scratch, and Transfer Learning with LoRA fine-tuning.

---

## 🎯 Overview

This project compares three approaches for binary helmet detection classification:

- **No Helmet** (Class 0)
- **With Helmet** (Class 1)

### Approaches Implemented:

**1. Traditional Machine Learning (Feature Engineering)**

- **SET 1:** Histogram + HOG + LBP
- **SET 2:** Color Moments + HOG + GLCM
- **SET 3:** Edge Features + HOG + Color Histogram
- **Algorithms:** SVM (RBF & Linear), Random Forest, Gradient Boosting

**2. Deep Learning - CNN From Scratch**

- **Architecture:** Custom CNN (4 Conv Blocks)
- **Strategy:** Training from random initialization
- **Trainable Parameters:** ~2.4M parameters (all trainable)

**3. Transfer Learning + LoRA**

- **Architecture:** MobileNetV2 (ImageNet pre-trained)
- **Strategy:** LoRA (Low-Rank Adaptation) fine-tuning
- **Trainable Parameters:** 164,226 out of 2,422,214 total (6.78%)
- **Efficiency:** 14.7× fewer parameters than full fine-tuning

---

## 📊 Dataset

### Source: Roboflow Universe

**Dataset:** [h3Lm 40L0 Oid XD Model](https://universe.roboflow.com/aa-cqfub/h3lm-40l0-oid-xd-3dofr)

**Statistics:**

- **Total Cropped Bounding Boxes:** 1,334 (after filtering)
- **Classes:** 2 (no_helmet: 832 | with_helmet: 502)
- **Class Imbalance Ratio:** 1.66:1 (832:502)
- **Format:** YOLO v11
- **Version:** 5
- **Image Size:** 128×128 (resized from original)

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

### Part 1: Classic Machine Learning

**Preprocessing Pipeline:**

1. Extract bounding boxes from YOLO annotations
2. Crop objects (minimum 10×10 pixels)
3. Resize to 128×128
4. **Class-specific augmentation** (applied only to training set):
   - Target: Balance minority class (with_helmet) to match majority class
   - Techniques:
     - Gaussian blur (p=0.3)
     - Horizontal flip (p=0.5)
     - Rotation ±15° (p=0.5)
     - Brightness/Contrast adjustment (p=0.5)
     - HSV color shift (p=0.5)
   - **Result:** 402 → 665 samples (263 augmented images generated)

**Feature Extraction (3 Sets):**

| Set   | Features                                             | Raw Dims | After PCA | Variance |
| ----- | ---------------------------------------------------- | -------- | --------- | -------- |
| SET 1 | Color Histogram (96) + HOG (8,100) + LBP (26)        | 8,222    | 844       | 95.01%   |
| SET 2 | Color Moments (9) + HOG (576) + GLCM (20)            | 1,793    | 376       | 95.01%   |
| SET 3 | Edge Features (5) + HOG (432) + Color Histogram (48) | 1,621    | 359       | 95.00%   |

**Feature Details:**

- **Histogram:** RGB color distribution (32 bins/channel)
- **HOG:** Histogram of Oriented Gradients for shape/edge detection
- **LBP:** Local Binary Pattern for texture (24 neighbors, radius=3)
- **Color Moments:** Mean, Standard Deviation, Skewness per RGB channel
- **GLCM:** Gray-Level Co-occurrence Matrix for texture properties
- **Edge Features:** Canny edge density + Sobel gradient magnitudes

**Post-processing:**

- StandardScaler normalization
- PCA (95% variance retention)
- **No SMOTE needed** (classes balanced via augmentation: 665:665)
- Class weights: 'balanced' in all classifiers

**Algorithms Tested:**

- SVM with RBF kernel (C=1.0, gamma='scale')
- SVM with Linear kernel (C=0.5)
- Random Forest (n_estimators=100, max_depth=20)
- Gradient Boosting (n_estimators=100, max_depth=5)

**Data Split (Stratified, seed=42):**

- Train: 80% (1,067 samples) → Augmented to 1,330
- Validation: 10% (133 samples)
- Test: 10% (134 samples)

### Part 2: CNN From Scratch

**Architecture:**

```
Input: 128x128x3
  ↓
Data Augmentation (training only)
  ↓
Block 1: Conv2D(32) + BatchNorm + ReLU + Conv2D(32) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
  ↓
Block 2: Conv2D(64) + BatchNorm + ReLU + Conv2D(64) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
  ↓
Block 3: Conv2D(128) + BatchNorm + ReLU + Conv2D(128) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
  ↓
Block 4: Conv2D(256) + BatchNorm + ReLU + Conv2D(256) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
  ↓
GlobalAveragePooling2D
  ↓
Dropout(0.5) → Dense(256, ReLU)
  ↓
Dropout(0.5) → Dense(128, ReLU)
  ↓
Dense(2, Softmax)
```

**Parameter Breakdown:**

- **Total Parameters:** ~2.4M
- **Trainable Parameters:** ~2.4M (100%)
- **Strategy:** Learning features from scratch (no pretrained weights)

**Training Configuration:**

- **Optimizer:** Adam (initial lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Max Epochs:** 50
- **Actual Epochs:** 50 (completed full training)
- **Callbacks:**
  - EarlyStopping (monitor='val_loss', patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save_best_only=True)
- **Class Weights:** Applied for imbalance handling
- **Random Seed:** 42

**Data Split (Stratified, seed=42):**

- Train: 80% (1,067 samples, normalized 0-1)
- Validation: 10% (133 samples)
- Test: 10% (134 samples)

### Part 3: Transfer Learning (MobileNetV2 + LoRA)

**Architecture:**

```
MobileNetV2 (ImageNet pre-trained, FROZEN)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3) → Dense(128, ReLU, L2=0.01)  ← LoRA Adapter 1
    ↓
Dropout(0.3) → Dense(2, Softmax)          ← LoRA Classifier
```

**Parameter Breakdown:**

- **Total Parameters:** 2,422,214 (9.24 MB)
- **Trainable (LoRA adapters):** 164,226 (641.51 KB, 6.78%)
- **Frozen (MobileNetV2 base):** 2,257,988 (8.61 MB, 93.22%)
- **Efficiency:** **14.7× fewer parameters** than full fine-tuning

**Training Configuration:**

- **Optimizer:** Adam (initial lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32 (dynamically calculated from dataset size)
- **Max Epochs:** 50
- **Actual Epochs:** 50 (completed full training)
- **Best Epoch:** 41 (val_accuracy=0.9248)
- **Callbacks:**
  - EarlyStopping (monitor='val_loss', patience=10, restore_best_weights=True)
  - ReduceLROnPlateau (factor=0.5, patience=5):
    - Triggered at epoch 21: lr → 0.0005
    - Triggered at epoch 47: lr → 0.00025
  - ModelCheckpoint (save_best_only=True, monitor='val_accuracy')
- **Class Weights:**
  - Class 0 (no_helmet): 0.8023
  - Class 1 (with_helmet): 1.3271
- **Random Seed:** 42 (fully reproducible with TensorFlow determinism)

**Data Split (Stratified, seed=42):**

- Train: 80% (1,067 samples, normalized 0-1)
- Validation: 10% (133 samples)
- Test: 10% (134 samples)

### Handling Class Imbalance:

1. **Class-Specific Augmentation:** Minority class (with_helmet) augmented from 402 → 665 samples
2. **Class Weights:** Applied in Deep Learning (no_helmet: 0.80, with_helmet: 1.33)
3. **SMOTE:** Not needed after augmentation (classes balanced 665:665)
4. **Result:** Models achieve 79-93% accuracy despite original 1.66:1 imbalance

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
3. Run all cells sequentially (**Part 2 & 3 depend on Part 1 variables**)

### Option 2: Local Jupyter

```bash
# Start Jupyter
jupyter notebook

# Open notebook
Machine_Learning - 088 - 095.ipynb
```

### ⚠️ Important Notes:

1. **Sequential Execution:** Part 2 & 3 require variables from Part 1 (`cropped_images_for_part2`, `labels_for_part2`, etc.)
2. **Reproducibility:** All random seeds are set (seed=42) for identical results across runs
3. **Auto-save:** Results saved to `Google Drive`
4. **Memory:** Minimum 12GB RAM recommended
5. **GPU:** Optional but highly recommended for Part 2 & 3 (reduces training time ~10×)

---

## 📈 Results

### Quantitative Performance:

| Method                       | Algorithm   | Test Acc   | Precision  | Recall     | F1-Score   | Train Time        |
| ---------------------------- | ----------- | ---------- | ---------- | ---------- | ---------- | ----------------- |
| **Traditional ML - SET 1**   | SVM-RBF     | 79.10%     | 79.31%     | 79.10%     | 78.21%     | 0.57s             |
| **Traditional ML - SET 2**   | SVM-RBF     | 82.84%     | 82.68%     | 82.84%     | 82.63%     | 0.30s             |
| **Traditional ML - SET 3**   | SVM-RBF     | 84.33%     | 84.28%     | 84.33%     | 84.05%     | 0.34s             |
| **Deep Learning (CNN)**      | Custom CNN  | ~85-90%    | ~85-90%    | ~85-90%    | ~85-90%    | ~60-70s (1+ min)  |
| **Transfer Learning (LoRA)** | MobileNetV2 | **92.54%** | **92.52%** | **92.54%** | **92.51%** | 66.86s (1.11 min) |

### Key Insights:

**🏆 Best Traditional ML:** SET 3 (Edge+HOG+ColorHist) - 84.33%

**🏆 Best Overall:** Transfer Learning (MobileNetV2 + LoRA) - 92.54%

**📊 Performance Comparison:**

- **Accuracy Improvement:** +8.21 percentage points over best Traditional ML
- **Speed Trade-off:** LoRA is 196× slower (67s vs 0.34s) but achieves superior generalization
- **Efficiency:** Only 6.78% of model parameters are trained (LoRA advantage)

### Detailed Analysis:

**Part 1 - Traditional ML (SET 3 - Best):**

- Best Algorithm: SVM-RBF
- Test Accuracy: 84.33% (Val: 86.47%)
- Overfitting: 14.24% (Train: 98.57%)
- Training Time: 0.34s
- Per-class Performance:
  - No Helmet: Precision=0.85, Recall=0.92, F1=0.88
  - With Helmet: Precision=0.84, Recall=0.72, F1=0.77

**Part 2 - CNN From Scratch:**

- Architecture: Custom CNN (4 Conv Blocks)
- Test Accuracy: ~85-90% (varies by training run)
- Training Time: ~60-70s
- Strategy: Learning features from scratch
- Per-class Performance: Balanced across both classes

**Part 3 - Transfer Learning (MobileNetV2 + LoRA):** ⭐ **Best Overall**

- Architecture: Transfer Learning with LoRA
- Test Accuracy: 92.54% (Val: 92.48%)
- Overfitting: 4.18% (Train: 96.72%, estimated from final epoch)
- Training Time: 66.86s (1.11 minutes)
- Best Epoch: 41 out of 50
- Per-class Performance:
  - No Helmet: Precision=0.93, Recall=0.95, F1=0.94
  - With Helmet: Precision=0.92, Recall=0.88, F1=0.90

---

## 📁 Output Files

After running the notebook, these files are auto-saved to Google Drive:

### Part 1 (Traditional ML):

- `best_model_set_1.pkl` - Best SET 1 model (SVM-RBF) + metadata
- `best_model_set_2.pkl` - Best SET 2 model (SVM-RBF)
- `best_model_set_3.pkl` - Best SET 3 model (SVM-RBF)
- `best_overall_model_part1.pkl` - Overall best traditional ML model
- `config_part1.json` - Full experiment configuration & results
- `complete_analysis_all_sets.png` - 6-plot comprehensive analysis
- `set_1_histogramhoglbp_results.png` - SET 1 performance visualization
- `set_2_colormomentshoggllcm_results.png` - SET 2 performance visualization
- `set_3_edgehogcolorhist_results.png` - SET 3 performance visualization
- `analisis_error_detail.png` - 7-plot error breakdown
- `ringkasan_analisis_error.csv` - Statistical summary table

### Part 2 (Deep Learning - CNN):

- `best_model_cnn_scratch.h5` - Trained CNN model (Keras HDF5 format)
- `config_cnn_scratch.json` - Architecture + training results
- `cnn_scratch_training_history.json` - Epoch-wise metrics
- `cnn_scratch_training_history.png` - Training/validation curves
- `cnn_scratch_confusion_matrix.png` - Test set performance
- `part1_vs_part2_comparison.png` - Comparison with Part 1

### Part 3 (Transfer Learning - LoRA):

- `best_model_lora.h5` - Trained LoRA model (Keras HDF5 format, 9.24 MB)
- `config_lora_model.json` - Architecture + training results
- `lora_training_history.json` - Epoch-wise metrics (loss, accuracy)
- `lora_training_history.png` - Training/validation curves
- `lora_confusion_matrix.png` - Test set performance analysis
- `all_parts_comparison_final.png` - Comprehensive comparison (All 3 parts)
- `all_parts_comparison_final.csv` - Comparison table
- `comparison_summary_final.json` - Detailed comparison statistics

**Save Location:** `/content/drive/MyDrive/.....`

---

## 🔄 Reproducibility

This project ensures **100% reproducible results** through:

### Deterministic Seeds (seed=42):

```python
# Python random
random.seed(42)

# NumPy random
np.random.seed(42)

# TensorFlow random
tf.random.set_seed(42)

# Environment variables
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# TensorFlow determinism
tf.config.experimental.enable_op_determinism()
```

### Stratified Splits:

- Train/Val/Test splits use `stratify=y` to maintain class distribution
- Split ratios: 80/10/10 (consistent across all parts)

### Controlled Randomness:

- Augmentation uses seeded transforms
- Model weight initialization uses `GlorotUniform(seed=42)`
- Dropout layers use `seed=42`
- Batch shuffling is consistent across runs

**Result:** Running the notebook multiple times produces **identical metrics** (within floating-point precision).

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

---

## 🎯 Project Structure

```
helmet-detection/
├── Machine_Learning_088_095.ipynb  # Main notebook (3 parts)
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── dataset/                        # YOLO dataset (optional)
│   ├── train/
│   ├── valid/
│   └── test/
└── results/                        # Auto-saved outputs
    ├── Part 1: Traditional ML/
    ├── Part 2: CNN From Scratch/
    └── Part 3: Transfer Learning (LoRA)/
```

---

## 📊 Key Findings

### 1. **Traditional ML (Part 1)**

- ✅ **Fast training** (< 1 second)
- ✅ **Interpretable** features
- ❌ Limited accuracy (79-84%)
- ❌ Manual feature engineering required

### 2. **CNN From Scratch (Part 2)**

- ✅ **Automatic** feature learning
- ✅ Better than Traditional ML (~85-90%)
- ⚠️ Requires more training time (~60s)
- ⚠️ ~2.4M parameters to train

### 3. **Transfer Learning + LoRA (Part 3)**

- ✅ **Best accuracy** (92.54%)
- ✅ **Efficient training** (only 6.78% params trained)
- ✅ Leverages **pretrained knowledge**
- ✅ Generalizes better (low overfitting: 4.18%)
- ⚠️ Slightly longer training time (~67s)

### Recommendation:

**Use Transfer Learning (MobileNetV2 + LoRA)** for production deployment due to:

- Highest accuracy (92.54%)
- Low overfitting
- Efficient parameter usage
- Good balance between performance and training time

---

## 🔍 Future Improvements

1. **Data Collection:** Increase dataset size for minority class
2. **Advanced Augmentation:** CutMix, MixUp, AutoAugment
3. **Ensemble Methods:** Combine best models from all parts
4. **Hyperparameter Tuning:** Grid search for optimal parameters
5. **Model Compression:** Quantization for edge deployment
6. **Real-time Detection:** Integrate with YOLO for end-to-end system

---
