import os
import cv2
import numpy as np
from ultralytics import SAM
from glob import glob
from tqdm import tqdm
import random
import shutil

# Paths
dataset_dir = "/Users/ahmad.aldarderi/Desktop/P&G/Line 10 VISADP4 Dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# YOLO dataset structure
train_images = os.path.join(images_dir, "train")
val_images = os.path.join(images_dir, "val")
train_labels = os.path.join(labels_dir, "train")
val_labels = os.path.join(labels_dir, "val")

os.makedirs(train_images, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Find all bounding boxes using fast OpenCV CV2 contours
def get_cv2_box(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 100)
    
    # morphological close to connect edges
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    
    if cv2.contourArea(c) > 500: # ensure it's substantial
        x, y, w, h = cv2.boundingRect(c)
        return np.array([[x, y, x+w, y+h]])
    return None

# Find all raw images in root
all_images = glob(os.path.join(dataset_dir, "*.jpg"))
random.shuffle(all_images)

# 80/20 Split
split_idx = int(len(all_images) * 0.8)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

print("Loading SAM for Auto-Annotation...")
sam_model = SAM('models/mobile_sam.pt')

def process_image(img_path, dest_images_dir, dest_labels_dir):
    filename = os.path.basename(img_path)
    label_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
    dest_img_path = os.path.join(dest_images_dir, filename)
    dest_label_path = os.path.join(dest_labels_dir, label_filename)
    
    try:
        # 1. CV2 BBox detection
        boxes = get_cv2_box(img_path)
        
        if boxes is None or len(boxes) == 0:
            shutil.move(img_path, dest_img_path)
            open(dest_label_path, 'w').close()
            return
            
        # 2. SAM Segmentation
        sam_results = sam_model(img_path, bboxes=boxes, verbose=False)
        masks = sam_results[0].masks
        h, w = sam_results[0].orig_shape
        
        # 3. Save as YOLO format
        with open(dest_label_path, 'w') as f:
            if masks is not None and masks.xy is not None:
                for segment in masks.xy:
                    if len(segment) == 0:
                        continue
                    segment_norm = segment.copy()
                    segment_norm[:, 0] /= w
                    segment_norm[:, 1] /= h
                    points = segment_norm.flatten().tolist()
                    line = "0 " + " ".join([f"{p:.6f}" for p in points])
                    f.write(line + "\n")
                    
        shutil.move(img_path, dest_img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if len(all_images) > 0:
    print(f"Annotating {len(train_files)} training images...")
    for img_path in tqdm(train_files):
        process_image(img_path, train_images, train_labels)
        
    print(f"Annotating {len(val_files)} validation images...")
    for img_path in tqdm(val_files):
        process_image(img_path, val_images, val_labels)

    # Generate dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: box\n")

    print("\n✅ Auto-annotation complete! Everything is perfectly prepared.")
else:
    print("\nImages have already been moved to the train/val directories, or no images found in dataset folder.")
    print("If you want to train immediately, run 'python train.py --data \"/Users/ahmad.aldarderi/Desktop/P&G/Line 10 VISADP4 Dataset/dataset.yaml\" --no-roboflow'")
