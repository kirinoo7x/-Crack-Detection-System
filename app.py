import streamlit as st
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------
REPO_ID = "watthanakrit/crack-models"
DEVICE = torch.device("cpu")

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    # download weights
    faster_path = hf_hub_download(REPO_ID, "faster_rcnn_finetune.pth")
    yolo_path   = hf_hub_download(REPO_ID, "best.pt")
    sam_path    = hf_hub_download(REPO_ID, "sam2_ft_final100.pt")

    # -----------------------
    # Faster R-CNN (FIX สำคัญ)
    # -----------------------
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    faster_model = fasterrcnn_resnet50_fpn(
        weights=None,
        num_classes=2   # ต้องตรงตอน train
    )

    state_dict = torch.load(faster_path, map_location=DEVICE)

    # รองรับทั้ง 2 format
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        faster_model.load_state_dict(state_dict["model_state_dict"])
    else:
        faster_model.load_state_dict(state_dict)

    faster_model.to(DEVICE)
    faster_model.eval()

    # -----------------------
    # YOLO
    # -----------------------
    yolo_model = YOLO(yolo_path)

    # -----------------------
    # SAM2 (กันพัง)
    # -----------------------
    try:
        sam_model = torch.load(sam_path, map_location=DEVICE)
        sam_model.to(DEVICE)
        sam_model.eval()
    except:
        sam_model = None

    return faster_model, yolo_model, sam_model


faster_model, yolo_model, sam_model = load_models()

# -----------------------
# UI
# -----------------------
st.title("🔍 Crack Detection System")

model_choice = st.selectbox(
    "Choose Model",
    ["Faster R-CNN", "YOLO", "SAM2 (Segmentation)"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -----------------------
# PREPROCESS (เอา cv2 ออก)
# -----------------------
def preprocess(img):
    return np.array(img)

# -----------------------
# INFERENCE
# -----------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = preprocess(img)

    # -----------------------
    # Faster R-CNN
    # -----------------------
    if model_choice == "Faster R-CNN":
        st.write("Running Faster R-CNN...")

        x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255
        x = x.to(DEVICE)

        outputs = faster_model([x])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        vis = img_np.copy()

        for box, score in zip(boxes, scores):
            if score > 0.7:
                x1, y1, x2, y2 = map(int, box)

                # วาด box ด้วย numpy (ไม่ใช้ cv2)
                vis[y1:y1+2, x1:x2] = [0,255,0]
                vis[y2-2:y2, x1:x2] = [0,255,0]
                vis[y1:y2, x1:x1+2] = [0,255,0]
                vis[y1:y2, x2-2:x2] = [0,255,0]

        st.image(vis)

    # -----------------------
    # YOLO
    # -----------------------
    elif model_choice == "YOLO":
        st.write("Running YOLO...")

        results = yolo_model(img_np)
        img_out = results[0].plot()

        st.image(img_out)

    # -----------------------
    # SAM2
    # -----------------------
    elif model_choice == "SAM2 (Segmentation)":
        st.write("Running SAM2...")

        if sam_model is None:
            st.error("❌ SAM2 โหลดไม่สำเร็จ")
            st.stop()

        x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255
        x = x.unsqueeze(0).to(DEVICE)

        try:
            outputs = sam_model(x)
        except:
            st.error("❌ SAM2 forward ไม่ได้ (model structure mismatch)")
            st.stop()

        mask = outputs.squeeze().detach().cpu().numpy()
        mask = (mask > 0.5)

        overlay = img_np.copy()
        overlay[mask] = [255, 0, 0]

        st.image(overlay)
