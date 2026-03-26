from ultralytics import YOLO
import os

print("\n--- Loading Custom Trained YOLOv8 Segmentation Model ---")
weights_path = 'models/pampers_custom_best.pt'

if not os.path.exists(weights_path):
    print(f"Error: Model not found at '{weights_path}'")
    print("Please wait for the training script to finish creating 'best.pt'!")
    exit(1)

model = YOLO(weights_path)

print("\n--- Running Validation on 94 Auto-Annotated Images ---")
# Evaluate model performance on the validation set using our built dataset
metrics = model.val(data='/Users/ahmad.aldarderi/Desktop/P&G/Line 10 VISADP4 Dataset/dataset.yaml')

print("\n================ FINAL ACCURACY METRICS ================")
print(f"Mask Segmentation Accuracy (mAP50-95):  {metrics.seg.map:.3f}")
print(f"Mask Segmentation Accuracy (mAP50):     {metrics.seg.map50:.3f}")
print("--------------------------------------------------------")
print("NOTE: 'mAP' goes from 0.0 to 1.0 (Higher is perfectly accurate).")
print("Detailed charts, curves, and validation images are saved in the new folder under 'runs/segment/val'")
print("========================================================\n")
