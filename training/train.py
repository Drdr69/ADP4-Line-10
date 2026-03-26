import argparse
import os
from ultralytics import YOLO
from roboflow import Roboflow

def train_model(data_yaml=None, epochs=100, imgsz=640, batch_size=32, use_roboflow=True):
    """
    Train a YOLOv8 instance segmentation model on custom data.
    Instance segmentation is ideal for irregular shapes as it provides a pixel-perfect mask.
    """
    # ------------------ ROBOFLOW CONFIG ------------------
    rf_api_key = "G2L5hUacsoNrdMHxCWVK"
    rf_workspace = "ahmads-workspace-1dj1i"
    rf_project = "actual-project-lr45w"  
    rf_version = 1                   
    # -----------------------------------------------------
    
    dataset_location = data_yaml
    
    # 1. Download Dataset from Roboflow
    if use_roboflow and not data_yaml:
        print("\n--- Downloading Dataset from Roboflow ---")
        rf = Roboflow(api_key=rf_api_key)
        project = rf.workspace(rf_workspace).project(rf_project)
        try:
            version = project.version(rf_version)
            dataset = version.download("yolov8")
            dataset_location = os.path.join(dataset.location, "data.yaml")
            print(f"Dataset downloaded successfully to: {dataset_location}")
        except Exception as e:
            print(f"Error downloading from Roboflow: {e}")
            print("Did you click 'Generate' in Roboflow to create Version 1?")
            return
            
    elif not data_yaml:
        print("Error: Please provide --data <path> or use Roboflow automation.")
        return

    # 2. Train YOLOv8
    print(f"\n--- Loading pre-trained YOLOv8 segmentation model (Nano for Speed) ---")
    model = YOLO('models/yolov8n-seg.pt')

    print(f"\n--- Starting training for {epochs} epochs on dataset: {dataset_location} ---")
    results = model.train(
        data=dataset_location,
        epochs=epochs,
        time=4.0, # 4 hours time limit
        imgsz=imgsz,
        batch=batch_size,
        device='mps', # Force Mac M-series GPU for massive speedup
        project='box_measurement',
        name='segmentation_model',
        exist_ok=True,
        # Robust augmentation for missing textures / super blank targets
        erasing=0.4,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0
    )
    print(f"\n--- Training Complete ---")
    
    # 3. Upload weights back to Roboflow
    if use_roboflow:
        print("\n--- Uploading weights to Roboflow ---")
        try:
            rf = Roboflow(api_key=rf_api_key)
            workspace = rf.workspace(rf_workspace)
            
            # YOLOv8 saves custom trained models here: project/name
            model_dir = "box_measurement/segmentation_model"
            
            workspace.deploy_model(
                model_type="yolov8",
                model_path=model_dir,
                project_ids=[rf_project],
                model_name="camera-readings-model", # Any name you like
                filename="weights/best.pt"
            )
            print("✅ Weights successfully uploaded to Roboflow!")
        except Exception as e:
            print(f"Error uploading to Roboflow: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model for Irregular Boxes")
    parser.add_argument("--data", type=str, default=None, help="Path to the dataset YAML file (leave blank to use Roboflow)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (lowered to 8 for yolov8x-seg.pt to avoid OOM)")
    parser.add_argument("--no-roboflow", action="store_true", help="Disable automatic Roboflow download/upload")
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.imgsz, args.batch, use_roboflow=not args.no_roboflow)
