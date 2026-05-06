import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    # Faster R-CNN
    faster_model = torch.load("faster_rcnn_finetune.pth", map_location=device)
    faster_model.eval()

    # YOLO
    from ultralytics import YOLO
    yolo_model = YOLO("best.pt")

    # SAM2 (feature + head)
    sam_model = torch.load("sam2_ft_final100.pth", map_location=device)
    sam_model.eval()

    return faster_model, yolo_model, sam_model


faster_model, yolo_model, sam_model = load_models()

# -----------------------
# UI
# -----------------------
st.title("🔍 Crack Detection System")
st.write("Upload image and choose model")

model_choice = st.selectbox(
    "Choose Model",
    ["Faster R-CNN", "YOLO", "SAM2 (Segmentation)"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

# -----------------------
# PROCESS IMAGE
# -----------------------
def preprocess(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# -----------------------
# INFERENCE
# -----------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img_cv = preprocess(img)

    if model_choice == "Faster R-CNN":
        st.write("Running Faster R-CNN...")
        x = torch.from_numpy(img_cv).permute(2,0,1).float()/255
        x = [x]

        outputs = faster_model(x)[0]

        boxes = outputs["boxes"].detach().numpy()
        scores = outputs["scores"].detach().numpy()

        for box, score in zip(boxes, scores):
            if score > 0.7:
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(img_cv, (x1,y1),(x2,y2),(0,255,0),2)

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    elif model_choice == "YOLO":
        st.write("Running YOLO...")
        results = yolo_model(img_cv)

        img_out = results[0].plot()
        st.image(img_out)

    elif model_choice == "SAM2 (Segmentation)":
        st.write("Running SAM2...")
        x = torch.from_numpy(img_cv).permute(2,0,1).float()/255
        x = x.unsqueeze(0)

        outputs = sam_model(x)
        mask = outputs.squeeze().detach().numpy()

        mask = (mask > 0.5).astype(np.uint8)*255

        # overlay
        overlay = img_cv.copy()
        overlay[mask==255] = [0,0,255]

        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
