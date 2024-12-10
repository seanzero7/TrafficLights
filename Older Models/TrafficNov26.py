import os
import json
import random
import shutil
import warnings
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Suppress warnings and model output
warnings.filterwarnings("ignore")
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Base path
base_path = r'C:\Users\Sean\Desktop\TrafficLights'

# Define paths
dataset_dir = os.path.join(base_path, 'train_dataset')
images_dir = os.path.join(dataset_dir, 'train_images')
json_file = os.path.join(dataset_dir, 'train.json')
output_dir = os.path.join(base_path, 'ProcessedTrafficLights')

# Create necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

# Load annotations
with open(json_file, 'r') as f:
    data = json.load(f)

annotations = data['annotations']

# Group annotations by filename
from collections import defaultdict

file_annotations = defaultdict(list)
for ann in annotations:
    filename = os.path.basename(ann['filename']).replace('\\', '/')
    file_annotations[filename].append(ann)

# List of all image filenames
all_images = list(file_annotations.keys())

# Split data into train and val
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.75)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

# Scaling function
def scale_bbox(bbox, img_width, img_height):
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    xmax = bbox['xmax']
    ymax = bbox['ymax']

    x_expand = (xmax - xmin) * 5
    y_expand = (ymax - ymin) * 5

    xmin_new = max(0, xmin - x_expand / 2)
    ymin_new = max(0, ymin - y_expand / 2)
    xmax_new = min(img_width, xmax + x_expand / 2)
    ymax_new = min(img_height, ymax + y_expand / 2)

    return xmin_new, ymin_new, xmax_new, ymax_new

# Function to convert bbox to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2]) / 2 / img_width
    y_center = (bbox[1] + bbox[3]) / 2 / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return x_center, y_center, width, height

# Process and save annotations
def process_data(files, subset):
    for filename in files:
        anns = file_annotations[filename]
        img_path = os.path.join(images_dir, filename)

        # Handle missing images
        if not os.path.exists(img_path):
            print(f"Image {filename} not found, skipping.")
            continue

        # Read image to get dimensions
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Copy image to output directory
        shutil.copy(img_path, os.path.join(output_dir, 'images', subset, filename))

        # Prepare label file
        label_lines = []
        for ann in anns:
            bbox = ann['bndbox']
            # Apply scaling
            xmin_new, ymin_new, xmax_new, ymax_new = scale_bbox(bbox, img_width, img_height)
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                [xmin_new, ymin_new, xmax_new, ymax_new], img_width, img_height
            )
            # For this example, we'll use class '0' for all traffic lights
            label_lines.append(f"0 {x_center} {y_center} {width} {height}")

        # Save label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', subset, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

# Process training and validation data
process_data(train_files, 'train')
process_data(val_files, 'val')

# Prepare YAML file for YOLO training
yaml_content = rf"""path: {output_dir}
train: images/train
val: images/val

names:
  0: traffic_light
"""

yaml_path = os.path.join(output_dir, 'traffic_light.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# Suppress Ultralytics output
from contextlib import redirect_stdout
import io

# Train YOLO model
model = YOLO('yolo11n.pt')  # Load the YOLO model 

results = model.train(data=yaml_path, epochs=10, imgsz=640)

# Function to compute MSE between predicted and ground truth bounding boxes
def compute_mse(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return None  # Cannot compute MSE if there are no predictions or ground truths
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)
    # For simplicity, match boxes by order
    min_len = min(len(pred_boxes), len(gt_boxes))
    pred_boxes = pred_boxes[:min_len]
    gt_boxes = gt_boxes[:min_len]
    mse = np.mean((pred_boxes - gt_boxes) ** 2)
    return mse

# Function to get ground truth boxes
def get_ground_truth_boxes(files, subset):
    gt_boxes_dict = {}
    for filename in files:
        anns = file_annotations[filename]
        img_path = os.path.join(images_dir, filename)

        # Handle missing images
        if not os.path.exists(img_path):
            continue

        # Read image to get dimensions
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        gt_boxes = []
        for ann in anns:
            bbox = ann['bndbox']
            xmin_new, ymin_new, xmax_new, ymax_new = scale_bbox(bbox, img_width, img_height)
            gt_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
        gt_boxes_dict[filename] = gt_boxes
    return gt_boxes_dict

# Get ground truth boxes for train and val
train_gt_boxes = get_ground_truth_boxes(train_files, 'train')
val_gt_boxes = get_ground_truth_boxes(val_files, 'val')

# Evaluate the model on training data
train_predictions = {}
train_mse_list = []
for filename in train_files:
    img_path = os.path.join(output_dir, 'images', 'train', filename)
    img = cv2.imread(img_path)
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Predicted bounding boxes
    train_predictions[filename] = boxes.tolist()
    gt_boxes = train_gt_boxes.get(filename, [])
    mse = compute_mse(boxes.tolist(), gt_boxes)
    if mse is not None:
        train_mse_list.append(mse)

train_mse = np.mean(train_mse_list) if train_mse_list else None

# Evaluate the model on validation data
val_predictions = {}
val_mse_list = []
for filename in val_files:
    img_path = os.path.join(output_dir, 'images', 'val', filename)
    img = cv2.imread(img_path)
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Predicted bounding boxes
    val_predictions[filename] = boxes.tolist()
    gt_boxes = val_gt_boxes.get(filename, [])
    mse = compute_mse(boxes.tolist(), gt_boxes)
    if mse is not None:
        val_mse_list.append(mse)

val_mse = np.mean(val_mse_list) if val_mse_list else None

# Store results in a dictionary
results_dict = {
    'train': {
        'predictions': train_predictions,
        'mse': train_mse,
    },
    'validation': {
        'predictions': val_predictions,
        'mse': val_mse,
    }
}

# Only print Training and Validation MSE
print(f"Training MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
