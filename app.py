import cv2
import numpy as np
from PIL import Image
import streamlit as st

def intelligent_score_contour(cnt, img_area, hierarchy, idx):
    area = cv2.contourArea(cnt)
    if area < 10:
        return 0
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return 0
    solidity = area / hull_area
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    rel_area = area / img_area
    is_hole = hierarchy[0][idx][3] != -1

    score = 0
    if rel_area > 0.0003:
        score += 1
    if 0.20 < solidity <= 1.0:
        score += 1
    if 0.1 < aspect_ratio < 10:
        score += 1
    if is_hole:
        score += 1
    return score

st.set_page_config(page_title="Contour Counter AI", layout="centered")
st.markdown("""
# Closed Contour Counter  
<small>By: Jacob Brown</small>  
<small>Date: 05/20/2025</small>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a logo image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Fixed threshold optimized for all logos
    color_diff_thresh = 25

    # Load image and extract background color
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)
    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    bg_color = img_rgb[0, 0].astype(int)

    # Compute color distance
    diff = np.abs(img_rgb.astype(int) - bg_color)
    color_distance = np.linalg.norm(diff, axis=2)

    # Create mask for foreground
    mask = (color_distance > color_diff_thresh).astype(np.uint8) * 255
    thresh = 255 - mask  # Invert for contour detection

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    selected = [
        cnt for i, cnt in enumerate(contours)
        if intelligent_score_contour(cnt, img_area, hierarchy, i) >= 3
    ]

    # Draw and display
    preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, selected, -1, (0, 255, 0), 1)

    st.image(pil_img, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected # of closed contours: {len(selected)}")
    st.image(preview, caption="Filtered Contours", channels="BGR", use_container_width=True)
