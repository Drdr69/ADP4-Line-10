import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

def load_config(config_path="calibration_config.json"):
    """
    Load pixel-to-millimeter ratio from calibration config.
    """
    if not os.path.exists(config_path):
        print("Warning: 'calibration_config.json' not found. We will only output Pixel dimensions.")
        return None, None, None
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("pixels_per_mm", None), config.get("pixels_per_mm_w", None), config.get("pixels_per_mm_h", None)

def load_camera_matrix(matrix_path="camera_matrix.json"):
    """
    Loads intrinsic distortion parameters. If missing, assumes no undistortion.
    """
    if not os.path.exists(matrix_path):
        return None, None
    with open(matrix_path, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"])

def undistort_image(img, mtx, dist):
    """
    Mathematically flattens the barrel/fish-eye effect of the lens to guarantee
    dimension accuracy across the entire FOV edges.
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # Crop the image to completely eliminate warped black borders
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

def process_mask_and_draw(result_img, largest_contour, pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h):
    """
    Given a raw contour, calculates minAreaRect dimensions in mm and draws boundaries onto the image.
    Returns the measurement dict containing exact data.
    """
    # cv2.minAreaRect finds the smallest possible rotated rectangle enclosing the contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    pixel_width, pixel_height = rect[1]
    
    measurements = {"pixel_width": pixel_width, "pixel_height": pixel_height, "unit": "px"}
    
    # --- PIXEL TO MILLIMETER CONVERSION ---
    if pixels_per_mm is not None and pixels_per_mm_w is not None and pixels_per_mm_h is not None:
        longest_side = max(pixel_width, pixel_height)
        shortest_side = min(pixel_width, pixel_height)
        
        width_mm = longest_side / pixels_per_mm_w
        height_mm = shortest_side / pixels_per_mm_h
        
        measurements.update({"width": width_mm, "height": height_mm, "unit": "mm"})
        
        text_w = f"Width: {width_mm:.2f}mm"
        text_h = f"Height: {height_mm:.2f}mm"
        print(f"Calculated Width: {width_mm:.2f} mm")
        print(f"Calculated Height: {height_mm:.2f} mm")
    else:
        # Fallback if only pixels_per_mm is set
        if pixels_per_mm:
            width_mm = pixel_width / pixels_per_mm
            height_mm = pixel_height / pixels_per_mm
            measurements.update({"width": width_mm, "height": height_mm, "unit": "mm"})
            text_w = f"Width: {width_mm:.2f}mm"
            text_h = f"Height: {height_mm:.2f}mm"
        else:
            text_w = f"Pixel Width: {pixel_width:.2f}px"
            text_h = f"Pixel Height: {pixel_height:.2f}px"
    
    # --- Visualization ---
    cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
    
    cv2.putText(result_img, text_w, (int(box[0][0]), int(box[0][1] - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result_img, text_h, (int(box[0][0]), int(box[0][1] - 35)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
    return measurements

def measure_box(image_path, model_path, calibration_path="calibration_config.json", output_dir="results"):
    """
    Main legacy testing function using the unified pipeline.
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h = load_config(calibration_path)
    mtx, dist = load_camera_matrix()
    
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # --- NEW UNDISTORTION PASS ---
    if mtx is not None and dist is not None:
        print("Applying Camera Mathematical Undistortion...")
        img = undistort_image(img, mtx, dist)
    
    results = model(img, conf=0.01)
    
    if len(results[0].boxes) == 0 or results[0].masks is None:
        print("No valid objects/masks detected.")
        return
        
    masks_data = results[0].masks.data.cpu().numpy()
    best_contours = None
    max_area = -1
    
    for m in masks_data:
        m_resized = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        b_mask = (m_resized * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                best_contours = cnts
                
    contours = best_contours
    
    if not contours:
        print("No valid contour found for the box mask.")
        return
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    result_img = img.copy()
    process_mask_and_draw(result_img, largest_contour, pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"measured_{base_name}")
    cv2.imwrite(output_path, result_img)
    print(f"\nSaved measurement result to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Measure irregular boxes.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="models/pampers_custom_best.pt", help="Path to model")
    parser.add_argument("--config", type=str, default="calibration_config.json", help="Path to config")
    args = parser.parse_args()
    measure_box(args.image, args.model, args.config)
