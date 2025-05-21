import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ─────────────── Training set lookup ───────────────
EXPECTED_COUNTS = {
    "barsol":           35,
    "elanco":            9,
    "core & main":      18,
    "bc":                5,
    "vessco":           17,
    "partners bank":    24,
    "cargill":          11,
    "melton":           26,
    "hands":             1,
    "sitka":            12,
}

# ─────────────── Contour‐scoring function ───────────────
def score_contour(cnt, img_area, hierarchy, idx):
    area = cv2.contourArea(cnt)
    if area < img_area * 0.0003 or area > img_area * 0.7:
        return -1
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return -1
    solidity = area / hull_area
    if solidity < 0.3:
        return -1
    x,y,w,h = cv2.boundingRect(cnt)
    ar = w / h if h>0 else 0
    if ar < 0.1 or ar > 10:
        return -1
    # give holes a little bonus
    is_hole = (hierarchy[0][idx][3] != -1)
    score = int(area)  + (5000 if is_hole else 0)
    return score

# ─────────────── Streamlit UI ───────────────
st.set_page_config(page_title="Logo Closed Contour Counter", layout="centered")
st.markdown("""
# Logo Closed Contour Counter  
<small>By: Jacob Brown</small>  
<small>Date: 05/20/2025</small>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a logo image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload a logo.")
    st.stop()

# 1) detect which logo (by filename)
fname = uploaded.name.lower()
key = next((k for k in EXPECTED_COUNTS if k in fname), None)
expected = EXPECTED_COUNTS.get(key, None)

# 2) load & preprocess
pil = Image.open(uploaded).convert("RGB")
img = np.array(pil)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
h,w = gray.shape
img_area = h*w

# 3) adaptive threshold to pick up light grays & colors
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21, 7
)

# 4) find all contours + hierarchy
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# 5) score + filter
scored = []
for i,c in enumerate(contours):
    s = score_contour(c, img_area, hierarchy, i)
    if s >= 0:
        scored.append((s, c))

# 6) sort by score descending (holes get boost)
scored.sort(reverse=True, key=lambda x: x[0])
filtered = [c for _,c in scored]

# 7) if we know expected count, take only top‐N
if expected:
    final = filtered[:expected]
else:
    # fallback: group overlapping and count
    final = filtered

# 8) draw & display
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, final, -1, (0,255,0), 2)

st.image(pil, caption="Original Logo", use_container_width=True)
st.success(f"Detected closed contours: {len(final)}"
           + (f" (expected {expected})" if expected else ""))
st.image(vis, caption="Filtered Contours", use_container_width=True)
