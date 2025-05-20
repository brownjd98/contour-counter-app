import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter (Include Holes & Inner Parts)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Load image and grayscale
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    h, w = img.shape
    img_area = h * w

    # Threshold
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Find all contours (including holes and children)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.002 or area > img_area * 0.7:
            continue  # remove noise or giant background shapes

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        ar = w_box / h_box
        if ar < 0.15 or ar > 10:
            continue  # ignore strange skinny fragments

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.3:
            continue  # exclude wobbly noise

        filtered_contours.append(cnt)

    # Display output
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours (filtered): {len(filtered_contours)}")

    with st.expander("Contour details"):
        for i, cnt in enumerate(filtered_contours, 1):
            st.write(f"#{i}: Area = {cv2.contourArea(cnt):.0f}")

    # Draw contours
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, filtered_contours, -1, (0, 255, 0), 2)
    st.image(preview, caption="Filtered Contours", channels="BGR", use_container_width=True)
