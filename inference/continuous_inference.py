import cv2
import numpy as np
import argparse
import os
import json
import time
import glob
import sys

# Ensure Python can find the 'core' folder regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.measurement_utils import load_config, load_camera_matrix, undistort_image, process_mask_and_draw
from ultralytics import YOLO

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def upload_data(image, measurement_data):
    """
    Simulates uploading the processed image and measurement data.
    """
    output_dir = "uploaded_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = int(time.time() * 1000)
    image_filename = os.path.join(output_dir, f"result_{timestamp}.jpg")
    json_filename = os.path.join(output_dir, f"data_{timestamp}.json")

    cv2.imwrite(image_filename, image)
    with open(json_filename, "w") as f:
        json.dump(measurement_data, f, indent=4)

    print(f"Uploaded! Saved image to {image_filename} and data to {json_filename}")

# ==============================================================================
# MAIN PROCESSING LOGIC
# ==============================================================================

def process_and_upload_frame(frame, model, pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h, mtx=None, dist=None):
    """
    Takes a single frame, perfectly flattens it with undistortion, detects masks, 
    and uses the core measurement utils to extract measurements mathematically.
    """
    if mtx is not None and dist is not None:
        frame = undistort_image(frame, mtx, dist)

    results = model(frame, verbose=False) 
    
    measurement_data = {
        "boxes_detected": 0,
        "measurements": []
    }
    
    result_img = frame.copy()
    
    if len(results[0].boxes) > 0:
        measurement_data["boxes_detected"] = len(results[0].boxes)
        masks_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        
        for idx, mask in enumerate(masks_data):
            # Stretch mask to original image bounds
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            # --- CENTRALISED MEASUREMENT AND DRAWING ---
            metrics = process_mask_and_draw(result_img, largest_contour, pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h)
            
            measurement_data["measurements"].append({
                "id": idx,
                "width": metrics.get("width", metrics.get("pixel_width")),
                "height": metrics.get("height", metrics.get("pixel_height")),
                "unit": metrics.get("unit")
            })
            
    upload_data(image=result_img, measurement_data=measurement_data)
    return result_img

# ==============================================================================
# THE INFINITE LOOP
# ==============================================================================

def continuous_inference(model_path="models/yolov8x-seg.pt", calibration_path="calibration_config.json", source="0"):
    print(f"Loading AI Model from {model_path}...")
    model = YOLO(model_path)
    
    pixels_per_mm, pixels_w, pixels_h = load_config(calibration_path)
    # Automatically load the distortion matrix using the central util
    mtx, dist = load_camera_matrix("camera_matrix.json")
    
    if source.isdigit():
        is_camera = True
        cap = cv2.VideoCapture(int(source))
    else:
        is_camera = False
        processed_images = set()

    print("\n--- Starting Continuous Program ---")
    
    try:
        while True:
            if is_camera:
                success, frame = cap.read()
                if not success:
                    time.sleep(1)
                    continue
                result_img = process_and_upload_frame(frame, model, pixels_per_mm, pixels_w, pixels_h, mtx, dist)
                cv2.imshow("Live Camera Feed", result_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                search_pattern = os.path.join(source, "*.*")
                all_files = glob.glob(search_pattern)
                new_images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f not in processed_images]
                
                for img_path in new_images:
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        result_img = process_and_upload_frame(frame, model, pixels_per_mm, pixels_w, pixels_h, mtx, dist)
                        cv2.imshow("Folder Feed", result_img)
                        cv2.waitKey(1)
                        processed_images.add(img_path)
                
                time.sleep(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if is_camera: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous measurement pipeline. Reads from Webcam or a Folder.")
    parser.add_argument("--model", type=str, default="models/pampers_custom_best.pt", help="Path to the trained YOLO model")
    parser.add_argument("--config", type=str, default="calibration_config.json", help="Path to calibration config JSON")
    parser.add_argument("--source", type=str, default="0", help="Camera ID or Folder Path")
    
    args = parser.parse_args()
    continuous_inference(args.model, args.config, args.source)
