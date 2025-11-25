# Helmet Detection: Classic ML vs CNN vs Vision Transformer (ViT + LoRA)

Comparative study of helmet detection using three approaches: Traditional Machine Learning (Feature Engineering), Deep Learning from Scratch, and Transfer Learning with Vision Transformer using LoRA fine-tuning.

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
- **Trainable Parameters:** 1,275,042 parameters (100% trainable)

**3. Transfer Learning - Vision Transformer (ViT + LoRA)**

- **Architecture:** ViT-Base/16 (ImageNet-21k pre-trained)
- **Strategy:** LoRA (Low-Rank Adaptation) fine-tuning
- **Trainable Parameters:** 1,538 out of 85,800,194 total (0.002%)
- **Efficiency:** 55,787× fewer parameters than full fine-tuning

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
| SET 1 | Color Histogram (96) + HOG (8,100) + LBP (26)        | 8,222    | 841       | 95.00%   |
| SET 2 | Color Moments (9) + HOG (576) + GLCM (20)            | 1,793    | 376       | 95.00%   |
| SET 3 | Edge Features (5) + HOG (432) + Color Histogram (48) | 1,621    | 362       | 95.03%   |

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

### Part 3: Transfer Learning (Vision Transformer + LoRA)

**Architecture:**

```
ViT-Base/16 (ImageNet-21k pre-trained, FROZEN)
  ↓
12 Transformer Encoder Layers (768 hidden size, 12 attention heads)
  ↓
Classification Head (TRAINABLE via LoRA):
  Dense(768 → 2, Softmax)
```

**Parameter Breakdown:**

- **Total Parameters:** 85,800,194 (327 MB)
- **Trainable (LoRA classifier):** 1,538 (6 KB, 0.002%)
- **Frozen (ViT encoder):** 85,798,656 (327 MB, 99.998%)
- **Efficiency:** **55,787× fewer parameters** than full fine-tuning

**Training Configuration:**

- **Optimizer:** AdamW (initial lr=0.0002)
- **Loss:** CrossEntropyLoss with class weights
- **Batch Size:** 16
- **Max Epochs:** 30
- **Actual Epochs:** 15 (early stopped)
- **Best Epoch:** 5 (val_accuracy=0.9624)
- **Callbacks:**
  - EarlyStopping (monitor='val_loss', patience=10, restore_best_weights=True)
  - ReduceLROnPlateau (factor=0.5, patience=5)
- **Class Weights:**
  - Class 0 (no_helmet): 0.8023
  - Class 1 (with_helmet): 1.3271
- **Random Seed:** 42 (fully reproducible with PyTorch determinism)

**Data Augmentation (training only):**

- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Random affine transform
- Color jitter (brightness, contrast, saturation, hue)

**Data Split (Stratified, seed=42):**

- Train: 80% (1,067 samples, resized to 224×224)
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
# Part 1 & 2 (TensorFlow-based)
pip install roboflow albumentations opencv-python scikit-learn scikit-image
pip install matplotlib seaborn imbalanced-learn tensorflow>=2.13.0
pip install pandas numpy jupyter

# Part 3 (PyTorch-based)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow datasets accelerate
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

| Method                           | Algorithm   | Test Acc   | Precision  | Recall     | F1-Score   | Train Time        |
| -------------------------------- | ----------- | ---------- | ---------- | ---------- | ---------- | ----------------- |
| **Traditional ML - SET 1**       | SVM-RBF     | 79.85%     | 80.69%     | 79.85%     | 78.71%     | 0.86s             |
| **Traditional ML - SET 2**       | SVM-RBF     | **82.84%** | 82.69%     | 82.84%     | 82.72%     | 0.24s             |
| **Traditional ML - SET 3**       | SVM-RBF     | **82.84%** | 82.69%     | 82.84%     | 82.72%     | 0.27s             |
| **Deep Learning (CNN)**          | Custom CNN  | **90.30%** | 90.77%     | 90.30%     | 90.07%     | 272.26s (4.5 min) |
| **Transfer Learning (ViT+LoRA)** | ViT-Base/16 | **94.78%** | **94.93%** | **94.78%** | **94.80%** | 248.97s (4.2 min) |

### Key Insights:

**🏆 Best Traditional ML:** SET 2 & SET 3 (tied) - 82.84%

**🏆 Best Overall:** Transfer Learning (Vision Transformer + LoRA) - 94.78%

**📊 Performance Comparison:**

- **Accuracy Improvement:** +11.94 percentage points over best Traditional ML
- **ViT vs CNN:** +4.48 percentage points improvement
- **Efficiency:** ViT uses only 0.002% trainable parameters (55,787× more efficient than full fine-tuning)

### Detailed Analysis:

**Part 1 - Traditional ML:**

**SET 1 (Histogram + HOG + LBP):**

- Best Algorithm: SVM-RBF
- Test Accuracy: 79.85% (Val: 83.46%)
- Overfitting: 19.17% (Train: 99.02%)
- Training Time: 0.86s
- PCA: 8,222 → 841 features (9.78× compression)
- Per-class Performance:
  - No Helmet: Precision=0.78, Recall=0.94, F1=0.85
  - With Helmet: Precision=0.85, Recall=0.56, F1=0.67

**SET 2 (Color Moments + HOG + GLCM):** ⭐ **Best Traditional ML (tied)**

- Best Algorithm: SVM-RBF
- Test Accuracy: 82.84% (Val: 87.22%)
- Overfitting: 15.81% (Train: 98.65%)
- Training Time: 0.24s
- PCA: 1,793 → 376 features (4.77× compression)
- Per-class Performance:
  - No Helmet: Precision=0.85, Recall=0.88, F1=0.87
  - With Helmet: Precision=0.79, Recall=0.74, F1=0.76

**SET 3 (Edge + HOG + Color Histogram):** ⭐ **Best Traditional ML (tied)**

- Best Algorithm: SVM-RBF
- Test Accuracy: 82.84% (Val: 85.71%)
- Overfitting: 15.89% (Train: 98.72%)
- Training Time: 0.27s
- PCA: 1,621 → 362 features (4.48× compression)
- Per-class Performance:
  - No Helmet: Precision=0.85, Recall=0.88, F1=0.87
  - With Helmet: Precision=0.79, Recall=0.74, F1=0.76

**Part 2 - CNN From Scratch:**

- Architecture: Custom CNN (4 Conv Blocks: 32→64→128→256 filters)
- Test Accuracy: 90.30% (Val: 90.23%)
- Overfitting: Low (~6% estimated)
- Training Time: 272.26s (4.54 minutes)
- Best Epoch: 36 out of 50 (early stopped at 47)
- Parameters: 1,275,042 (all trainable, 4.86 MB)
- Per-class Performance:
  - No Helmet: Precision=0.88, Recall=0.98, F1=0.93
  - With Helmet: Precision=0.95, Recall=0.78, F1=0.86

**Part 3 - Transfer Learning (Vision Transformer + LoRA):** ⭐ **Best Overall**

- Architecture: ViT-Base/16 with LoRA fine-tuning
- Test Accuracy: 94.78% (Val: 96.24% at best epoch)
- Overfitting: Very low (train ~94%, test ~95%)
- Training Time: 248.97s (4.15 minutes)
- Best Epoch: 5 out of 30 (early stopped at 15)
- Parameters: 85,800,194 total, **only 1,538 trainable** (0.002%)
- Per-class Performance:
  - No Helmet: Precision=0.98, Recall=0.94, F1=0.96
  - With Helmet: Precision=0.91, Recall=0.96, F1=0.93

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

### Part 3 (Transfer Learning - ViT):

- `best_vit_model.pth` - Trained Vision Transformer model (PyTorch format, 327 MB)
- `config_vit_model.json` - Architecture + training results
- `vit_training_history.json` - Epoch-wise metrics (loss, accuracy)
- `vit_training_history.png` - Training/validation curves
- `vit_confusion_matrix.png` - Test set performance analysis
- `vit_error_analysis.png` - Detailed error breakdown
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

# TensorFlow random (Part 1 & 2)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.experimental.enable_op_determinism()

# PyTorch random (Part 3)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
# Core dependencies
roboflow>=1.0.0
albumentations>=1.3.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
scikit-image>=0.20.0
matplotlib>=3.7.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0
pandas>=1.5.0
numpy>=1.24.0
jupyter>=1.0.0

# Part 1 & 2 (TensorFlow)
tensorflow>=2.13.0

# Part 3 (PyTorch + Transformers)
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
pillow>=9.0.0
datasets>=2.12.0
accelerate>=0.20.0
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

- ✅ **Very fast training** (< 1 second)
- ✅ **Interpretable** features
- ✅ **Low resource usage**
- ❌ Limited accuracy (79-83%)
- ❌ Manual feature engineering required
- 🏆 **Best SET:** SET 2 & SET 3 (tied at 82.84%)

### 2. **CNN From Scratch (Part 2)**

- ✅ **Automatic** feature learning
- ✅ Significantly better than Traditional ML (+7.46%)
- ✅ Reasonable training time (~4.5 minutes)
- ⚠️ 1.3M parameters to train
- ⚠️ Requires more data for optimal performance
- 🎯 **Accuracy:** 90.30%

### 3. **Transfer Learning - Vision Transformer (Part 3)**

- ✅ **Best accuracy** (94.78%)
- ✅ **Extremely efficient training** (only 1,538 params trained!)
- ✅ Leverages **pretrained knowledge** (ImageNet-21k)
- ✅ Generalizes better (lowest overfitting)
- ✅ Fast convergence (best at epoch 5)
- ⚠️ Requires more memory (327 MB model)
- 🏆 **Winner:** Best overall performance

### Performance Progression:

```
Traditional ML → CNN → Vision Transformer
   82.84%    →  90.30%  →    94.78%
  (+0% baseline) (+7.46%)   (+11.94%)
```

### Recommendation:

**Use Vision Transformer (ViT + LoRA)** for production deployment due to:

- Highest accuracy (94.78%)
- Extremely low overfitting
- Ultra-efficient parameter usage (0.002% trained)
- Fast training convergence
- State-of-the-art transformer architecture
- Good balance between performance and resource efficiency

---

## 🔍 Future Improvements

1. **Data Collection:** Increase dataset size, especially for minority class
2. **Advanced Augmentation:** CutMix, MixUp, AutoAugment
3. **Model Ensemble:** Combine CNN + ViT predictions
4. **Hyperparameter Tuning:** Optimize learning rate, batch size, etc.
5. **Full ViT Fine-tuning:** Compare LoRA vs full fine-tuning
6. **Larger ViT Models:** Test ViT-Large or ViT-Huge variants
7. **Other Transformers:** Try Swin Transformer, DeiT, BEiT
8. **Quantization:** INT8 quantization for edge deployment
9. **Real-time Detection:** Integrate with YOLO for end-to-end system
10. **Multi-task Learning:** Joint detection + classification

---

## 🙏 Acknowledgments

- Dataset: [Roboflow Universe](https://universe.roboflow.com/aa-cqfub/h3lm-40l0-oid-xd-3dofr)
- Pretrained Models:
  - Vision Transformer (ViT-Base/16) from Google Research
  - ImageNet-21k pre-training
- Frameworks:
  - TensorFlow/Keras (Part 1 & 2)
  - PyTorch + Hugging Face Transformers (Part 3)
  - scikit-learn, OpenCV
- LoRA: [Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2106.09685)
- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---
