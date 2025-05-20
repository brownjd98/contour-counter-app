import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Closed Contour Counter (Targeted 10-Shape Logic)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Load image and grayscale
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    h, w = img.shape
    img_area = h * w

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10
    )

    # Find all contours + hierarchy
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 1: Find the 3 biggest shapes (outer oval + 2 capsules)
    contour_data = [(i, cnt, cv2.contourArea(cnt)) for i, cnt in enumerate(contours)]
    sorted_by_area = sorted(contour_data, key=lambda x: x[2], reverse=True)
    main_shapes = [contours[i] for i, _, _ in sorted_by_area[:3]]

    # Step 2: Get 7 good holes (children with decent size/solidity)
    letter_holes = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue  # skip non-children
        area = cv2.contourArea(cnt)
        if not (img_area * 0.0005 < area < img_area * 0.05):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity > 0.4:
            letter_holes.append((cnt, area))

    # Sort holes by area and pick 7 best
    letter_holes = sorted(letter_holes, key=lambda x: x[1], reverse=True)
    hole_contours = [cnt for cnt, _ in letter_holes[:7]]

    # Final result
    final_contours = main_shapes + hole_contours

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success(f"Detected closed contours: {len(final_contours)}")

    with st.expander("Contour Details"):
        for i, cnt in enumerate(final_contours, 1):
            st.write(f"#{i}: Area = {cv2.contourArea(cnt):.0f}")

    # Draw
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, final_contours, -1, (0, 255, 0), 2)
    st.image(preview, caption="Targeted Contours", channels="BGR", use_container_width=True)
