import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_np = np.array(image)

    # ✅ Use adaptive thresholding for better shape detection
    thresh = cv2.adaptiveThreshold(
        img_np, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Find all contours including nested ones
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # Display original image and result
    st.image(image, caption="Uploaded Image", use_container_width=True)
    min_area = 200  # adjust this number as needed
    closed_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    st.success(f"Number of closed contours: {len(closed_contours)}")


    # Optional: Show contours overlay (helpful for visual debugging)
    preview = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, closed_contours, -1, (0, 255, 0), 1)
    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
