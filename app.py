import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_np = np.array(image)

    _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Number of closed contours: {len(contours)}")
