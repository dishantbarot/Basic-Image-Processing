import streamlit as st
import cv2
import numpy as np
from PIL import Image


# ---------- Utility functions ----------

def load_image_bgr(uploaded_file):
    """Load uploaded image as OpenCV BGR image."""
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def show_image_bgr(img_bgr, caption=None):
    """Display a BGR image in Streamlit as RGB."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, use_column_width=True)


def to_grayscale(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def get_properties(img_bgr):
    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1
    props = {
        "Width": w,
        "Height": h,
        "Channels": channels,
        "Shape": img_bgr.shape,
        "Data type": str(img_bgr.dtype),
        "Total pixels": img_bgr.size
    }
    return props


def rotate_image(img_bgr, angle):
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img_bgr


def mirror_image(img_bgr):
    return cv2.flip(img_bgr, 1)  # horizontal flip


def make_grid(img_bgr, rows=4, cols=4):
    """Make a grid over the image (non-prime number of cells e.g., 4x4=16)."""
    h, w = img_bgr.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    grid_img = img_bgr.copy()
    # Horizontal lines
    for r in range(1, rows):
        y = r * cell_h
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)
    # Vertical lines
    for c in range(1, cols):
        x = c * cell_w
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)

    return grid_img


def detect_objects(img_bgr, min_area=500):
    """Detect objects without deep learning using edges + contours."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_img = img_bgr.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(obj_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            count += 1

    return obj_img, count


# ---------- Streamlit App ----------

st.title("Basic Image Processing App 🧠📷")
st.write("Practical 1 – Deep Learning (Basic Image Processing)")

uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image as BGR (OpenCV)
    img_bgr = load_image_bgr(uploaded_file)

    # Create tabs for each operation
    tabs = st.tabs([
        "1) Show Image",
        "2) Color to Black & White",
        "3) Properties",
        "4) Rotate Image",
        "5) Mirror Image",
        "6) Make Small Grid",
        "7) Find Objects (No DL)",
        "8) Select All Options"
    ])

    # 1) Show image
    with tabs[0]:
        st.subheader("Original Image")
        show_image_bgr(img_bgr, caption="Original Image")

    # 2) Color to Black & White
    with tabs[1]:
        st.subheader("Black & White (Grayscale)")
        gray = to_grayscale(img_bgr)
        st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

    # 3) Properties
    with tabs[2]:
        st.subheader("Image Properties")
        props = get_properties(img_bgr)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

    # 4) Rotate Image
    with tabs[3]:
        st.subheader("Rotate the Image")
        angle = st.radio("Select rotation angle:", [90, 180, 270], index=0, horizontal=True)
        rotated = rotate_image(img_bgr, angle)
        show_image_bgr(rotated, caption=f"Rotated {angle} degrees")

    # 5) Mirror Image
    with tabs[4]:
        st.subheader("Mirror Image (Horizontal Flip)")
        mirrored = mirror_image(img_bgr)
        show_image_bgr(mirrored, caption="Mirror Image")

    # 6) Make Small Grid
    with tabs[5]:
        st.subheader("Grid on the Image (4x4 = 16 cells, non-prime)")
        grid_img = make_grid(img_bgr, rows=4, cols=4)
        show_image_bgr(grid_img, caption="Image with 4x4 Grid")

    # 7) Find Objects without Deep Learning
    with tabs[6]:
        st.subheader("Object Detection (Edge + Contours)")
        obj_img, count = detect_objects(img_bgr)
        st.write(f"**Approximate number of detected objects:** {count}")
        show_image_bgr(obj_img, caption="Detected Objects (Bounding Boxes)")

    # 8) Select All Options
    with tabs[7]:
        st.subheader("All Operations Combined")

        st.write("### Original")
        show_image_bgr(img_bgr)

        st.write("### Black & White")
        st.image(to_grayscale(img_bgr), use_column_width=True, clamp=True)

        st.write("### Properties")
        props = get_properties(img_bgr)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

        st.write("### Rotated 90°, 180°, 270°")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_image_bgr(rotate_image(img_bgr, 90), caption="90°")
        with col2:
            show_image_bgr(rotate_image(img_bgr, 180), caption="180°")
        with col3:
            show_image_bgr(rotate_image(img_bgr, 270), caption="270°")

        st.write("### Mirror Image")
        show_image_bgr(mirror_image(img_bgr), caption="Mirror Image")

        st.write("### Grid (4x4)")
        show_image_bgr(make_grid(img_bgr, 4, 4), caption="4x4 Grid")

        st.write("### Object Detection")
        obj_img_all, count_all = detect_objects(img_bgr)
        st.write(f"**Objects detected:** {count_all}")
        show_image_bgr(obj_img_all, caption="Detected Objects")

else:
    st.info("👆 Please upload a JPG / JPEG / PNG image to begin.")
