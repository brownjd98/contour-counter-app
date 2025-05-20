import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_barsol(image):
    img = np.array(image.convert("L"))
    img_area = img.shape[0] * img.shape[1]
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 1: largest 3 exterior shapes (oval + capsules)
    outer_shapes = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1]
    outer_shapes = sorted(outer_shapes, key=cv2.contourArea, reverse=True)[:3]

    # Step 2: top 7 inner holes (letterform holes)
    holes = [
        cnt for i, cnt in enumerate(contours)
        if hierarchy[0][i][3] != -1 and img_area * 0.0002 < cv2.contourArea(cnt) < img_area * 0.06
    ]
    holes = sorted(holes, key=cv2.contourArea, reverse=True)[:7]

    final_contours = outer_shapes + holes
    return img, final_contours

def process_bc(image):
    img = np.array(image.convert("L"))
    img_area = img.shape[0] * img.shape[1]
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 1: Filter clean letter strokes
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (img_area * 0.01 < area < img_area * 0.9):
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.6:
            continue
        filtered.append(cnt)

    # Step 2: Group nearby strokes into B and C
    boxes = [cv2.boundingRect(c) for c in filtered]
    used = [False] * len(filtered)
    groups = []

    for i in range(len(filtered)):
        if used[i]: continue
        xi, yi, wi, hi = boxes[i]
        group = [i]
        for j in range(i + 1, len(filtered)):
            if used[j]: continue
            xj, yj, wj, hj = boxes[j]
            if abs(xi - xj) < 5 and abs(yi - yj) < 5 and abs(wi - wj) < 10 and abs(hi - hj) < 10:
                group.append(j)
                used[j] = True
        used[i] = True
        groups.append(group)

    final_contours = [filtered[g[0]] for g in groups]
    return img, final_contours

def process_default(image):
    img = np.array(image.convert("L"))
    img_area = img.shape[0] * img.shape[1]
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [
        cnt for cnt in contours
        if img_area * 0.0005 < cv2.contourArea(cnt) < img_area * 0.9
    ]
    return img, filtered

# ─────────────────────────────────────────────────────────────

st.title("Smart Closed Contour Counter")

uploaded = st.file_uploader("Upload a logo/image", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded)
    filename = uploaded.name.lower()

    if "barsol" in filename:
        img, contours = process_barsol(image)
    elif "bc" in filename or "screenshot" in filename:
        img, contours = process_bc(image)
    else:
        img, contours = process_default(image)

    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours: {len(contours)}")

    with st.expander("Contour Details"):
        for i, cnt in enumerate(contours, 1):
            st.write(f"#{i}: Area = {cv2.contourArea(cnt):.0f}")

    st.image(preview, caption="Detected Contours", channels="BGR", use_container_width=True)
