import cv2
import numpy as np
from PIL import Image
import streamlit as st

def group_contours(contours, proximity=15):
    """Group contours whose bounding boxes overlap or sit close together."""
    boxes = [cv2.boundingRect(c) for c in contours]
    used = [False]*len(contours)
    groups = []
    for i in range(len(contours)):
        if used[i]:
            continue
        xi, yi, wi, hi = boxes[i]
        grp = [i]
        used[i] = True
        for j in range(i+1, len(contours)):
            if used[j]:
                continue
            xj, yj, wj, hj = boxes[j]
            # if boxes overlap or within 'proximity' pixels
            if not (xi+wi+proximity < xj or xj+wj+proximity < xi or
                    yi+hi+proximity < yj or yj+hj+proximity < yi):
                grp.append(j)
                used[j] = True
        groups.append(grp)
    return groups

def intelligent_score_contour(cnt, img_area, hierarchy, idx):
    area = cv2.contourArea(cnt)
    if area < img_area * 0.0005:     # drop tiny bits
        return False
    if area > img_area * 0.7:        # drop huge background
        return False
    # must be relatively “solid”
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area <= 0 or area/hull_area < 0.4:
        return False
    # drop very elongated slivers
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w/h if h>0 else 0
    if ar < 0.1 or ar > 10:
        return False
    return True

# ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Contour Counter AI", layout="centered")
st.markdown("""
# Logo Closed Contour Counter  
<small>By: Jacob Brown</small>  
<small>Date: 05/20/2025</small>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a logo image", type=["jpg","jpeg","png"])
if uploaded:
    # 1) Load & convert
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    img_area = h*w

    # 2) Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=5
    )

    # 3) Find contours + hierarchy
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4) Filter by shape & area
    valid = []
    for i, cnt in enumerate(contours):
        if intelligent_score_contour(cnt, img_area, hierarchy, i):
            valid.append(cnt)

    # 5) Group nearby/overlapping contours
    groups = group_contours(valid, proximity=15)
    # one representative per group:
    final_contours = [ valid[g[0]] for g in groups ]

    # 6) Draw & display
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(display, final_contours, -1, (0,255,0), 2)

    st.image(pil, caption="Original Logo", use_container_width=True)
    st.success(f"Detected closed contours: {len(final_contours)}")
    st.image(display, caption="Filtered + Grouped Contours", use_container_width=True)
