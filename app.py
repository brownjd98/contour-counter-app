import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter (Smart Detection)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Find all contours (with holes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Smart filtering
    smart_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue  # too small

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if aspect_ratio > 5 or aspect_ratio < 0.1:
            continue  # too flat or too tall

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.4:
            continue  # very jagged

        smart_contours.append(cnt)

    # Show results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Smartly detected closed contours: {len(smart_contours)}")

    with st.expander("Show filtered contour areas"):
        for i, cnt in enumerate(smart_contours):
            area = cv2.contourArea(cnt)
            st.write(f"Contour {i + 1}: Area = {area:.2f}")

    preview = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, smart_contours, -1, (0, 255, 0), 1)
    st.image(preview, caption="Detected Smart Contours", channels="BGR", use_container_width=True)
