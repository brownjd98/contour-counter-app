import streamlit as st
import cv2
import numpy as np
from PIL import Image

def merge_overlapping(contours, overlap_thresh=0.2):
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    groups = []
    used = [False] * len(boxes)

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        group = [i]
        xi1, yi1, xi2, yi2 = x1, y1, x1+w1, y1+h1
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j or used[j]:
                continue
            xj1, yj1, xj2, yj2 = x2, y2, x2+w2, y2+h2
            xa1, ya1 = max(xi1, xj1), max(yi1, yj1)
            xa2, ya2 = min(xi2, xj2), min(yi2, yj2)
            inter_area = max(0, xa2 - xa1) * max(0, ya2 - ya1)
            union_area = w1*h1 + w2*h2 - inter_area
            iou = inter_area / float(union_area)
            if iou > overlap_thresh:
                group.append(j)
                used[j] = True
        used[i] = True
        groups.append(group)
    return groups

st.title("Smart Closed Contour Counter (Grouped)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    h, w = img.shape
    img_area = h * w

    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small and huge noise
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.001 or area > img_area * 0.7:
            continue
        valid_contours.append(cnt)

    # Merge overlapping contours to count grouped objects
    groups = merge_overlapping(valid_contours)

    # Draw one contour from each group
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for group in groups:
        cv2.drawContours(preview, [valid_contours[group[0]]], -1, (0, 255, 0), 2)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Estimated distinct closed shapes: {len(groups)}")

    st.image(preview, caption="Grouped Contours", channels="BGR", use_container_width=True)
