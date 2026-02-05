# INBreast Mass Detection - ResNet34 with Advanced Preprocessing

Implementation of a **deep learning system for detecting masses in mammography images** using ResNet34 with advanced preprocessing techniques (CLAHE + Unsharp Masking) and Focal Loss for handling class imbalance.

---

## Problem

Build a system that:
- detects **masses in mammography images** from the INBreast dataset
- handles **significant class imbalance** between normal and mass-containing images
- enhances **low-contrast medical images** for better feature extraction
- achieves **high F1 score** for clinical relevance
- prevents **data leakage** through patient-level splitting

---

## Solution Overview

- Fine-tuned **ResNet34** architecture with dropout regularization
- Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for local contrast enhancement
- Used **Unsharp Masking** for edge detection and mass boundary highlighting
- Implemented **Focal Loss** to address class imbalance
- Applied **extensive data augmentation** for better generalization
- Used **patient-level train/val/test splitting** to prevent data leakage

---

## Data

### Dataset Structure
- **INBreast dataset**  
→ Mammography images + Excel file with metadata and labels

- **Excel file contains**:
  - Patient ID
  - File Name
  - Laterality (L/R)
  - View (CC/MLO)
  - **Binary label: Mass (1) vs No Mass (0)**
    
### Training
- **INBreast dataset (train split)**  
→ Mammography images with binary labels (Mass vs No Mass)

### Validation & Testing
- **INBreast dataset (validation & test splits)**  
→ ensures fair evaluation on unseen, patient-disjoint data

---

## Model & Training

- **Model**: `ResNet34` (pretrained on ImageNet)
- **Task**: Binary classification (Mass detection)
- **Framework**: PyTorch + Albumentations + OpenCV
- **Training setup**:
  - Focal Loss with α=0.25, γ=2.0
  - Class weights for imbalance handling
  - AdamW optimizer with layer-wise learning rates
  - CosineAnnealingLR scheduler
  - Dropout (p=0.5) + gradient clipping
  - 80 epochs, batch size 32

---

## Evaluation & Testing

- **Validation Set (INBreast validation split)**  
  - Used during training to monitor model performance and select the best epoch  
  - Metric used: **F1 Score**  
    → model checkpoint with highest F1 score on validation set is saved as the best model

- **Test Set (INBreast test split)**  
  - Used for final evaluation after training  
  - Metrics computed: **Accuracy**, **F1 Score**, **Precision**, **Recall**  
    → provides a comprehensive assessment of mass detection performance

---

## Results

Achieved on held-out test set (32 images):

- **Test Accuracy**: 87.50%
- **Test F1 Score**: 80%

**Per-class performance:**
- No Mass: Precision=0.87, Recall=0.95, F1=0.91
- Has Mass: Precision=0.89, Recall=0.73, F1=0.80

The model demonstrates strong performance with high recall for normal cases (minimizing false alarms) and balanced precision across both classes.

---

## Preprocessing Techniques Comparison

We evaluated the impact of different preprocessing techniques on mammography images:

- **Preprocessing methods**:
  - Min-Max Normalization (baseline)
  - Histogram Equalization (global contrast)
  - CLAHE (local contrast enhancement)
  - CLAHE + Unsharp Mask (final pipeline)

- **Loss functions**:
  - Cross-Entropy with class weights
  - Focal Loss (α=0.25, γ=2.0)

- **Comparison Metrics**:
  - **F1 Score** – higher is better (balance of precision and recall)
  - **Accuracy** – higher is better (overall correctness)
  - **Precision** – higher is better (minimize false positives)
  - **Recall** – higher is better (minimize false negatives)

- **Findings**:
  - CLAHE significantly improved local contrast in low-contrast mammography images
  - Unsharp masking enhanced mass boundary detection
  - Focal Loss effectively handled class imbalance by focusing on hard examples
  - Patient-level splitting prevented data leakage and improved generalization
