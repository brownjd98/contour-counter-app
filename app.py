import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_np = np.array(image)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Always use RETR_TREE (all contours, including nested/holes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Suggest default min_area as 0.05% of total image area
    image_area = img_np.shape[0] * img_np.shape[1]
    default_min_area = int(image_area * 0.0005)

    # Slider for area filtering (0–15000)
    min_area = st.slider("Minimum contour area", min_value=0, max_value=15000, value=default_min_area)

    # Filter contours by area
    closed_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Display original image and result
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Number of closed contours (area > {min_area}): {len(closed_contours)}")

    # Optional: show individual areas
    with st.expander("Show contour areas"):
        for i, cnt in enumerate(closed_contours):
            area = cv2.contourArea(cnt)
            st.write(f"Contour {i + 1}: Area = {area:.2f}")

    # Draw contours on grayscale background
    preview = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, closed_contours, -1, (0, 255, 0), 1)
    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
