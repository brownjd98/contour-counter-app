import cv2
import numpy as np
from PIL import Image
import streamlit as st

def intelligent_score_contour(cnt, img_area, hierarchy, idx):
    area = cv2.contourArea(cnt)
    if area < 15:
        return 0  # eliminate tiny specks
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return 0
    solidity = area / hull_area
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    rel_area = area / img_area
    is_hole = hierarchy[0][idx][3] != -1

    score = 0
    if rel_area > 0.0004:
        score += 1
    if 0.25 < solidity <= 1.0:
        score += 1
    if 0.15 < aspect_ratio < 8:
        score += 1
    if is_hole:
        score += 1
    return score

# Streamlit UI
st.set_page_config(page_title="Intelligent Contour Detector", layout="centered")
st.title("🧠 Intelligent Logo Contour Counter")

uploaded_file = st.file_uploader("Upload a logo image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("L")
    img_np = np.array(pil_img)
    img_area = img_np.shape[0] * img_np.shape[1]

    # Binarize with margin for white backgrounds
    _, thresh = cv2.threshold(img_np, 180, 255, cv2.THRESH_BINARY_INV)

    # Get contours including nested ones
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    selected = []
    for i, cnt in enumerate(contours):
        if intelligent_score_contour(cnt, img_area, hierarchy, i) >= 3:
            selected.append(cnt)

    # Visualize
    preview = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, selected, -1, (0, 255, 0), 1)

    st.image(pil_img, caption="Uploaded Logo", use_container_width=True)
    st.success(f"Detected intelligent closed contours: {len(selected)}")
    st.image(preview, caption="Filtered Contours", channels="BGR", use_container_width=True)
