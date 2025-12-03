# 📷 Basic Image Processing App

A Streamlit-based interactive application built to demonstrate **basic image processing techniques** commonly used in Deep Learning, Computer Vision, and AI.  
This app enables users to upload an image and apply multiple transformations and analysis operations through an intuitive UI — all without requiring deep learning models.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 📤 Upload Image | Supports JPG / JPEG / PNG |
| 🖼 Show Image | Displays the original image in RGB |
| ⚫ Grayscale Conversion | Converts color image to black & white |
| 📐 Image Properties | Width, height, channels, shape, total pixels |
| 🔄 Rotate Image | Rotate 90°, 180°, 270° options |
| 🪞 Mirror Image | Horizontal flipping |
| 🔳 Grid Overlay | Adds a 4×4 non-prime number grid on image |
| 🔍 Object Detection | Basic contour-based object identification (no DL) |
| 🎛 Select All Options | Combined output for all processing features |

---

## 🧰 Tech Stack

- **Python**
- **Streamlit**
- **OpenCV (opencv-python-headless)**
- **NumPy**
- **Pillow (PIL)**

---

📁 Project Directory

│── basic_image_processing_app.py        # Main Streamlit application

│── requirements.txt                      # Python dependencies

│── runtime.txt                           # Python version (for Streamlit Cloud)

│── README.md                             # Documentation

└── .streamlit/
     └── config.toml                      # Force Light Theme


