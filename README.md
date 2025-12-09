# YOLO11 Training & Advanced Visualization Pipeline

This script provides a complete end-to-end pipeline for training a YOLO11 object detection model, validating its performance, and generating high-resolution, research-grade visualizations.

Unlike standard YOLO export tools, this script includes a custom visualization engine that overlays **Grid Separation**, **Ground Truth vs. Predictions**, and calculates per-box **Intersection over Union (IoU)** scores manually.

## 📋 Table of Contents

  * [Overview](https://www.google.com/search?q=%23overview)
  * [Key Features](https://www.google.com/search?q=%23key-features)
  * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
  * [Configuration](https://www.google.com/search?q=%23configuration)
  * [Visualization Details](https://www.google.com/search?q=%23visualization-details)
  * [Usage](https://www.google.com/search?q=%23usage)
  * [Recommended Dataset](https://www.google.com/search?q=%23usage)

-----

## Overview

This pipeline is designed for researchers who need granular insight into their model's performance. It automates the setup of the environment, trains the `yolo11n` model on a custom dataset, and outputs standard metrics (Precision, Recall, mAP).

Crucially, it concludes by taking a specific sample image and generating a "debug-style" plot that reveals exactly how the model's predictions align with the ground truth and the underlying neural network grid.

-----

## Key Features

### 1\. Automated Training & Validation

  * Auto-detects GPU/CPU availability.
  * Trains for a set number of epochs (default: 50).
  * Extracts and prints key research metrics: **Precision**, **Recall**, **mAP@50**, and **Peak F1-Score**.

### 2\. Custom IoU Calculation

The script includes a manual implementation of the Intersection over Union (IoU) formula to verify prediction accuracy against ground truth boxes mathematically.

$$IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

### 3\. Research-Grade Visualization

Generates a detailed output image containing:

  * **Grid Overlay:** A 32-pixel stride grid (standard for YOLO's deepest layer) to visualize spatial resolution.
  * **Ground Truth (Green):** The actual labeled objects.
  * **Predictions (Red):** The model's detections.
  * **Data Annotations:** Every prediction box is tagged with its Confidence Score and the calculated IoU.

-----

## Prerequisites

The script automatically installs the necessary libraries if they are missing:

  * `torch`, `torchvision`, `torchaudio` (CUDA 11.8 compatible)
  * `ultralytics` (YOLO11)
  * `opencv-python` (cv2)
  * `matplotlib`

-----

## Configuration

Before running the script, you **must** update the path placeholders to match your directory structure.

### 1\. Dataset Configuration

Ensure your `data.yaml` file is correctly formatted and referenced in the training block:

```python
results = model.train(
    data="/dir/path/subpath/data.yaml",  # <--- UPDATE THIS
    # ...
)
```

### 2\. Model Weights

After training, the script looks for the best weights. Ensure this path aligns with your training output directory:

```python
best_weight_path = "/dir/path/subpath/runs/detect/train/weights/best.pt" # <--- UPDATE THIS
```

### 3\. Sample Image for Visualization

To generate the detailed plot, point the script to a valid image and its corresponding label file from your validation or test set:

```python
sample_img_path = "/dir/path/subpath/valid/images/sample1.jpg" # <--- UPDATE THIS
sample_lbl_path = "/dir/path/subpath/valid/labels/sample1.txt" # <--- UPDATE THIS
```

-----

## Visualization Details

When the script runs the `plot_detailed_sample` function, it produces an image with specific visual indicators:

| Element | Color | Description |
| :--- | :--- | :--- |
| **Grid Lines** | ⚪ White (Dashed) | Represents the **32px stride**. Objects smaller than a grid cell are harder to detect. |
| **Ground Truth** | 🟢 Green | The manual annotations from your dataset. |
| **Prediction** | 🔴 Red | The bounding box predicted by YOLO11. |
| **Labels** | 🟥 Red Background | Contains **Conf** (Confidence) and **IoU** (Overlap accuracy). |

-----

## Usage

1.  **Mount Drive (if using Colab):** Ensure your dataset is accessible.
2.  **Update Paths:** Edit the file paths in the `Configuration` section of the script.
3.  **Run Script:** Execute the Python script.
4.  **View Output:**
      * Console: metrics (Precision, Recall, mAP).
      * File System: `detailed_output.png` will be saved to the current directory.

### Example Output Console

```text
Using device: 0 (Tesla T4)
...
=== FINAL METRICS FOR RESEARCH PAPER ===
Precision (PPV):   0.9210
Recall (TPR):      0.8950
Accuracy (mAP@50): 0.9420
✅ Detailed sample saved to: detailed_output.png
```

-----

## Recommended Dataset

[Traffic Signs Europe by "Radu Oprea"](https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-detection-europe) 

<img width="1202" height="442" alt="image" src="https://github.com/user-attachments/assets/55de507b-ec3c-43eb-8c7e-ea9b6aeee31c" />

