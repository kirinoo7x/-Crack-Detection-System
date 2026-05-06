import streamlit as st
import torch
import cv2
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
    faster_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="faster_rcnn_finetune.pth"
    )

    yolo_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="best.pt"
    )

    sam_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="sam2_ft_final100.pt"
    )

    # -----------------------
    # Faster R-CNN (สำคัญ: ต้อง rebuild model)
    # -----------------------
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    faster_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    faster_model.load_state_dict(torch.load(faster_path, map_location=DEVICE))
    faster_model.to(DEVICE)
    faster_model.eval()

    # -----------------------
    # YOLO
    # -----------------------
    yolo_model = YOLO(yolo_path)

    # -----------------------
    # SAM2 (โหลดตรง)
    # -----------------------
    sam_model = torch.load(sam_path, map_location=DEVICE)
    sam_model.to(DEVICE)
    sam_model.eval()

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
# PREPROCESS
# -----------------------
def preprocess(img):
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# -----------------------
# INFERENCE
# -----------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_cv = preprocess(img)

    # -----------------------
    # Faster R-CNN
    # -----------------------
    if model_choice == "Faster R-CNN":
        st.write("Running Faster R-CNN...")

        x = torch.from_numpy(img_cv).permute(2, 0, 1).float() / 255
        x = x.to(DEVICE)
        outputs = faster_model([x])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        vis = img_cv.copy()

        for box, score in zip(boxes, scores):
            if score > 0.7:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    # -----------------------
    # YOLO
    # -----------------------
    elif model_choice == "YOLO":
        st.write("Running YOLO...")

        results = yolo_model(img_cv)
        img_out = results[0].plot()

        st.image(img_out)

    # -----------------------
    # SAM2
    # -----------------------
    elif model_choice == "SAM2 (Segmentation)":
        st.write("Running SAM2...")

        x = torch.from_numpy(img_cv).permute(2, 0, 1).float() / 255
        x = x.unsqueeze(0).to(DEVICE)

        try:
            outputs = sam_model(x)
        except:
            st.error("❌ SAM2 model forward error (structure mismatch)")
            st.stop()

        mask = outputs.squeeze().detach().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        # overlay
        overlay = img_cv.copy()
        overlay[mask == 1] = [0, 0, 255]

        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
