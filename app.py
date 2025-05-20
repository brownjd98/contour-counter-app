import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter (Smart Detection)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Load image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    h, w = img.shape
    img_area = h * w

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Get all contours (including nested)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Smart filtering
    smart_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.001 or area > img_area * 0.7:
            continue  # ignore very small or huge shapes

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / h_box
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.3:
            continue

        smart_contours.append(cnt)

    # Display results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours (auto-filtered): {len(smart_contours)}")

    with st.expander("Contour details"):
        for i, cnt in enumerate(smart_contours, 1):
            st.write(f"#{i}: Area = {cv2.contourArea(cnt):.0f}")

    # Draw detected contours
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, smart_contours, -1, (0, 255, 0), 2)
    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
