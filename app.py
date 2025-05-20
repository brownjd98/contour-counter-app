import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    img_np = np.array(image)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Dropdown: contour retrieval mode
    mode_label = st.selectbox(
        "Contour Mode",
        ["External", "All (with holes)"]
    )
    retrieval_mode = cv2.RETR_EXTERNAL if mode_label == "External" else cv2.RETR_TREE

    # Find contours
    contours, _ = cv2.findContours(thresh, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    # Auto-suggest min_area as 0.05% of image area
    image_area = img_np.shape[0] * img_np.shape[1]
    default_min_area = int(image_area * 0.0005)  # 0.05%

    min_area = st.slider("Minimum contour area", min_value=0, max_value=5000, value=default_min_area)

    # Filter contours by area
    closed_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Display original image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Number of closed contours (area > {min_area}): {len(closed_contours)}")

    # Optional: list areas
    with st.expander("Show contour areas"):
        for i, cnt in enumerate(closed_contours):
            area = cv2.contourArea(cnt)
            st.write(f"Contour {i + 1}: Area = {area:.2f}")

    # Draw filtered contours
    preview = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, closed_contours, -1, (0, 255, 0), 1)
    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
