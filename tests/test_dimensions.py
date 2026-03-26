import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
from core.measurement_utils import measure_box

val_images = glob.glob("/Users/ahmad.aldarderi/Desktop/P&G/Line 10 VISADP4 Dataset/images/val/*.jpg")
model_path = "models/pampers_custom_best.pt"
output_dir = "measurement_tests"

os.makedirs(output_dir, exist_ok=True)

# Test on the first 5 images found
print("Running measurement test on 5 validation images...")
for img_path in val_images[:5]:
    measure_box(img_path, model_path, output_dir=output_dir)

print(f"\nMeasurements written to: {output_dir}")
os.system(f"open {output_dir}")
