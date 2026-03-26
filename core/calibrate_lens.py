import cv2
import numpy as np
import os
import glob
import json
import argparse

def calibrate_camera(image_dir, chessboard_size, square_size_mm, output_file="camera_matrix.json"):
    """
    Calculates the Intrinsic Camera Matrix and Distortion Coefficients (k1, k2, p1, p2, k3)
    from a folder of printed checkerboard images captured by your camera.
    """
    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, e.g., (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # The chessboard size is the number of INTERNAL corners e.g., (9,6)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm  # Scale by physical square dimension

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.*'))
    
    if not images:
        print(f"Error: No images found in directory '{image_dir}'")
        return

    print(f"Found {len(images)} images to process...")
    valid_images = 0

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images += 1
            
            # Optionally draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Detecting Checkerboard', img)
            cv2.waitKey(100)
        else:
            print(f"Warning: Could not find pure {chessboard_size} chessboard pattern in {fname}")

    cv2.destroyAllWindows()

    if valid_images == 0:
        print("\nERROR: Could not find valid chessboard patterns in any image!")
        print("Please ensure your chessboard matches the inner-corner parameters perfectly.")
        return
        
    print(f"\nSuccessfully extracted corners from {valid_images}/{len(images)} images. Calculating Camera Matrix...")

    # Calculate Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Mean error calculation
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        
    print(f"\n✅ Total Calibration Error: {mean_error/len(objpoints):.4f} pixels (closer to 0 is perfectly flat)")

    # Save to JSON
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "mean_error_px": mean_error/len(objpoints)
    }

    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"\n🔥 SUCCESS! Saved Distortion Matrix to '{output_file}'!")
    print("The system will now mathematically flatten all lenses automatically prior to inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Intrinsic Matrix from a Folder of Checkerboard Photos.")
    parser.add_argument("--images", type=str, required=True, help="Folder containing checkerboard photos.")
    parser.add_argument("--grid", type=int, nargs=2, default=[9, 6], help="Number of inner corners (width height). Default 9 6.")
    parser.add_argument("--size", type=float, default=25.0, help="Size of a single printed square in millimeters. Default 25.0.")
    parser.add_argument("--output", type=str, default="camera_matrix.json", help="Output JSON path. Default camera_matrix.json")
    
    args = parser.parse_args()
    calibrate_camera(args.images, tuple(args.grid), args.size, args.output)
