import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
from ultralytics import YOLO

# Ensure internal modules can be resolved
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.measurement_utils import load_config, load_camera_matrix, undistort_image, process_mask_and_draw

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Box AI | Intelligent Measurement",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful modern dashboard UI
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #00FFBB;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #A0AEC0;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CACHED LOADERS (Prevents extremely slow reloads on every button click)
# ==============================================================================
@st.cache_resource
def get_model(model_name):
    return YOLO(model_name)

@st.cache_data
def get_calibrations():
    pixels_per_mm, p_w, p_h = load_config("calibration_config.json")
    mtx, dist = load_camera_matrix("camera_matrix.json")
    return pixels_per_mm, p_w, p_h, mtx, dist

# ==============================================================================
# APPLICATION UI
# ==============================================================================

st.markdown('<p class="main-header">📦 Intelligent Dimension Extractor</p>', unsafe_allow_html=True)
st.markdown("Instantly process packages using the highly accurate core mathematical instance-segmentation pipeline.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    st.markdown("### 🧠 Neural Weights")
    st.success("✅ **Custom Model:** Pampers Instance Segmenter")
    st.caption("Active: `models/pampers_custom_best.pt`")
    selected_model = "models/pampers_custom_best.pt"
    
    st.divider()
    st.markdown("### 📷 System Status")
    
    pixels_per_mm, p_w, p_h, mtx, dist = get_calibrations()
    
    if mtx is not None:
        st.success("✅ **Lens Calibration:** Loaded")
        st.caption("Active: Hardware mathematical undistortion is running.")
    else:
        st.warning("⚠️ **Lens Curve Fix:** Missing")
        st.caption("Inactive: Run `python core/calibrate_lens.py --images [dir]` to fix lens stretching.")

# --- INPUT HANDLING ---
mode = st.radio("Select Input Method:", ["Upload Image 📁", "Use Web Camera 📷"], horizontal=True)

image_file = None
if "Upload Image" in mode:
    image_file = st.file_uploader("Upload an irregular box package for measurement...", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Take a picture of the target area")

if image_file is not None:
    # 1. Convert to OpenCV format safely
    pil_image = Image.open(image_file).convert("RGB")
    frame = np.array(pil_image)
    frame = frame[:, :, ::-1].copy() # RGB to BGR for OpenCV
    
    st.divider()
    st.subheader("Results Dashboard")
    
    col_orig, col_proc, col_metrics = st.columns([1.5, 1.5, 1])
    
    with col_orig:
        st.markdown("#### 📸 Original Image")
        st.image(pil_image, caption="Unprocessed Capture", use_container_width=True)
        
    with col_proc:
        st.markdown("#### 🧠 Processed Output")
        with st.spinner("🧠 Initializing Neural Engine & Math Wrappers..."):
            model = get_model(selected_model)
            
            # Application of the CORE mathematical unified features
            proc_frame = frame.copy()
            if mtx is not None and dist is not None:
                proc_frame = undistort_image(proc_frame, mtx, dist)
                
            results = model(proc_frame, verbose=False)
            
            if len(results[0].boxes) > 0 and results[0].masks is not None:
                
                # We need to explicitly find the LARGEST mask in the image, to prevent 
                # YOLO from randomly selecting tiny artifact boxes in the background.
                masks_data = results[0].masks.data.cpu().numpy()
                best_contours = None
                max_area = -1
                
                for m in masks_data:
                    mask_resized = cv2.resize(m, (proc_frame.shape[1], proc_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized * 255).astype(np.uint8)
                    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        if area > max_area:
                            max_area = area
                            best_contours = cnts
                
                contours = best_contours
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Extract dimensions and draw using the unified core utility
                    metrics = process_mask_and_draw(proc_frame, largest_contour, pixels_per_mm, p_w, p_h)
                    
                    # Display the final rendered frame in RGB
                    proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    st.image(proc_frame_rgb, caption="Neural Mathematical Render", use_container_width=True)
                    
                    with col_metrics:
                        st.markdown("### Exact Dimensions")
                        unit = metrics.get('unit', 'px')
                        w_val = metrics.get('width', metrics.get('pixel_width', 0))
                        h_val = metrics.get('height', metrics.get('pixel_height', 0))
                        
                        st.metric(label=f"Physical Width ({unit})", value=f"{w_val:.2f}")
                        st.metric(label=f"Physical Height ({unit})", value=f"{h_val:.2f}")
                        
                        st.divider()
                        st.info("💡 **Insight:** Minimum Area Rectangle calculated flawlessly enclosing the physical contour of the instance mask. Rotational placement variance neutralized.")
                        
                else:
                    st.warning("Mask resolved but logical contour could not be closed.")
            else:
                st.error("No packages detected in frame. Adjust lighting or model weights.")
