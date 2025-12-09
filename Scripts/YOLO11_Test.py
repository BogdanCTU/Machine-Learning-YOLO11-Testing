# Installation, comment if already installed
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install ultralytics

# Navigate to your working directory (optional but recommended)
%cd /dir/path/subpath   # sample 

# --- Import YOLO ---
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

#================================================================
# Train

# Auto-select device based on availability
device = 0 if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = YOLO("yolo11n.pt")

results = model.train(
          data="/dir/path/subpath/data.yaml",
          epochs=50,
          imgsz=320,
          batch=8,
          device=device,
          workers=8
)

#================================================================
# Validation

import glob
import os

# 3. THE LATEST MODEL
best_weight_path = "/dir/path/subpath/train/weights/best.pt"

# 4. LOAD THE TRAINED MODEL
model = YOLO(best_weight_path)

# Run validation to get the raw data
results = model.val(data="/dir/path/subpath/data.yaml")

#================================================================
# Output Metrics

# 1. Precision & Recall (These are averages at the best confidence threshold)
precision = results.box.mean_results()[0]
recall = results.box.mean_results()[1]
map50 = results.box.map50

# 2. F1-Score
# results.box.f1 is the Mean F1 Curve (shape: [1000]).
# We take the maximum value from this curve to find the "Best F1".
f1_curve = results.box.f1

print(f"\n=== FINAL METRICS FOR RESEARCH PAPER ===")
print(f"Precision (PPV):   {precision:.4f}")
print(f"Recall (TPR):      {recall:.4f}")
print(f"Accuracy (mAP@50): {map50:.4f}")

def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # The area of both rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # IoU = Intersection / Union
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def plot_detailed_sample(image_path, label_path, model, save_path="detailed_output.png"):
    # 1. Load Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    # 2. Get Predictions (YOLO)
    results = model(image_path, verbose=False)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
    confs = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy()
    # 3. Get Ground Truth (from .txt file)
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                # YOLO format is normalized [x_center, y_center, width, height]
                nx, ny, nw, nh = parts[1], parts[2], parts[3], parts[4]
                # Convert to [x1, y1, x2, y2] pixel coordinates
                x1 = int((nx - nw / 2) * w)
                y1 = int((ny - nh / 2) * h)
                x2 = int((nx + nw / 2) * w)
                y2 = int((ny + nh / 2) * h)
                gt_boxes.append([x1, y1, x2, y2, class_id])
    # 4. Setup Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    # --- A. DRAW GRID (Stride 32 is standard for YOLO's deepest layer) ---
    grid_stride = 32
    for x in range(0, w, grid_stride):
        ax.axvline(x, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    for y in range(0, h, grid_stride):
        ax.axhline(y, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    # --- B. DRAW GROUND TRUTH (Green) ---
    matched_gt_indices = set()
    for i, gt in enumerate(gt_boxes):
        gx1, gy1, gx2, gy2, g_cls = gt
        rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(gx1, gy1-5, "Ground Truth", color='lime', fontsize=8, backgroundcolor='black')
    # --- C. DRAW PREDICTIONS & IoU (Red) ---
    for i, pred in enumerate(pred_boxes):
        px1, py1, px2, py2 = pred
        # Calculate IoU with every GT box and find the best match
        best_iou = 0.0
        for gt in gt_boxes:
            current_iou = calculate_iou([px1, py1, px2, py2], gt[:4])
            if current_iou > best_iou:
                best_iou = current_iou
        # Draw Box
        rect = patches.Rectangle((px1, py1), px2-px1, py2-py1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Annotation Text: IoU, Confidence, Coords
        label_text = (
            f"Pred\n"
            f"Conf: {confs[i]:.2f}\n"
            f"IoU: {best_iou:.2f}"
        )
        ax.text(px1, py1-5, label_text, color='white', fontsize=7, 
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))
    # Clean up plot
    plt.axis('off')
    plt.title(f"YOLO11 Analysis: Grid (32px), GT (Green) vs Pred (Red) & IoU", fontsize=14)  
    # Save
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Detailed sample saved to: {save_path}")
    plt.show()

#================================================================
# Run on selectd sample

# Replace these paths with a REAL image and label from your validation set
sample_img_path = "/dir/path/subpath/valid/images/sampleiamge1.jpg" 
sample_lbl_path = "/dir/path/subpath/valid/labels/sampleiamge1.txt"

# Ensure the paths exist before running
if os.path.exists(sample_img_path):
    plot_detailed_sample(sample_img_path, sample_lbl_path, model)
else:
    print(f"Could not find image at {sample_img_path}. Please update the path.")
