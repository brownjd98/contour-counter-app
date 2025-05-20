import streamlit as st
import cv2
import numpy as np
from PIL import Image

def filter_contours(img):
    img_area = img.shape[0] * img.shape[1]
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.0005 or area > img_area * 0.9:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.3:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / h_box
        if aspect_ratio < 0.1 or aspect_ratio > 15:
            continue
        filtered.append(cnt)
    return filtered, thresh

def group_similar_contours(contours, proximity_thresh=10):
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    groups = []
    used = [False] * len(contours)

    for i in range(len(contours)):
        if used[i]:
            continue
        group = [i]
        x1, y1, w1, h1 = boxes[i]
        box1 = (x1, y1, x1 + w1, y1 + h1)

        for j in range(i + 1, len(contours)):
            if used[j]:
                continue
            x2, y2, w2, h2 = boxes[j]
            box2 = (x2, y2, x2 + w2, y2 + h2)

            if not (box1[2] + proximity_thresh < box2[0] or box1[0] - proximity_thresh > box2[2] or
                    box1[3] + proximity_thresh < box2[1] or box1[1] - proximity_thresh > box2[3]):
                group.append(j)
                used[j] = True

        used[i] = True
        groups.append(group)
    return groups

st.title("Accurate Closed Contour Counter")

uploaded_file = st.file_uploader("Upload a logo/image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    filename = uploaded_file.name.lower()

    # Filter contours
    filtered, _ = filter_contours(img)

    # Only apply grouping for BC-style simple logos
    if "bc" in filename:
        groups = group_similar_contours(filtered, proximity_thresh=10)
        final_count = len(groups)
        contours_to_draw = [filtered[g[0]] for g in groups]
    else:
        final_count = len(filtered)
        contours_to_draw = filtered

    # Draw
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, contours_to_draw, -1, (0, 255, 0), 2)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours: {final_count}")

    with st.expander("Contour Details"):
        for i, cnt in enumerate(contours_to_draw, 1):
            st.write(f"#{i}: Area = {cv2.contourArea(cnt):.0f}")

    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
