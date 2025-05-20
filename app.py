import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter (Auto-Filtered)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # 1) Load & grayscale
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    h, w = img.shape
    img_area = h * w

    # 2) Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # 3) Find ONLY external shapes (letters, panels, border)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 4) Smart filter by area + aspect ratio + solidity
    auto_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # ignore tiny specks or enormous background
        if area < img_area * 0.0005 or area > img_area * 0.8:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        ar = cw / ch
        # drop extremely skinny or flat fragments
        if ar < 0.2 or ar > 5.0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        # letters and panels are fairly solid
        if solidity < 0.4:
            continue

        auto_contours.append(cnt)

    # 5) Display results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Auto-detected closed contours: {len(auto_contours)}")

    # Optional: show each area for debug
    with st.expander("Contour details"):
        for i, cnt in enumerate(auto_contours, 1):
            st.write(f"#{i}: Area={cv2.contourArea(cnt):.0f}")

    # 6) Overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, auto_contours, -1, (0,255,0), 2)
    st.image(overlay, caption="Filtered Contours", channels="BGR", use_container_width=True)
