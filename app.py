import streamlit as st
import cv2
import numpy as np
from PIL import Image

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

            # Check if boxes intersect or are close
            if not (box1[2] + proximity_thresh < box2[0] or box1[0] - proximity_thresh > box2[2] or
                    box1[3] + proximity_thresh < box2[1] or box1[1] - proximity_thresh > box2[3]):
                group.append(j)
                used[j] = True

        used[i] = True
        groups.append(group)
    return groups

st.title("Intelligent Closed Contour Counter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Load and convert to grayscale
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    img_area = img.shape[0] * img.shape[1]

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter meaningful ones
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

    # Group overlapping contours
    grouped = group_similar_contours(filtered)

    # Visualize one contour per group
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for group in grouped:
        cv2.drawContours(preview, [filtered[group[0]]], -1, (0, 255, 0), 2)

    # Show original and result
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours (grouped): {len(grouped)}")

    with st.expander("Contour Group Details"):
        for i, group in enumerate(grouped, 1):
            total_area = sum([cv2.contourArea(filtered[j]) for j in group])
            st.write(f"Group #{i}: {len(group)} merged contours, total area = {total_area:.0f}")

    st.image(preview, caption="Grouped Visual Contours", channels="BGR", use_container_width=True)
