# Student ID Monitoring System

## Overview

This project is an Student ID Monitoring System developed using YOLOv8, OpenCV, and Streamlit.

The system detects students in real time and verifies whether they are wearing valid ID cards by checking the presence of:
- ID Card
- ID Strap

The application supports:
- Image-based detection
- Real-time webcam detection
- Multi-person verification
- Live detection statistics dashboard

---

## Features

- Image Detection
- Live Webcam Detection
- Multi-person Verification
- Streamlit Dashboard
- YOLOv8 Object Detection
- Detection Statistics
- Real-time Monitoring

---

## Technologies Used

- Python
- YOLOv8n (Nano Model)
- OpenCV
- Streamlit
- Ultralytics
- NumPy

---

## Model Performance

- Precision: 91.55% 
- Recall: 92.86% 
- F1 Score: 92.2% 
- mAP50: 86.39%

## Project Structure

```bash
id_card_detection/
│
├── models/
│   ├── best.pt
│   └── yolov8n.pt
│
├── scripts/
│   ├── image_inference.py
│   ├── realtime_clean_ui.py
│   └── train.py
│
├── train/
├── valid/
├── test/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data.yaml
```

## Deployment Note

The deployed Streamlit cloud version currently supports:
- Image-based detection

Live webcam detection works only on the local machine because OpenCV webcam access (`cv2.VideoCapture`) is not supported on Streamlit Cloud servers.

To use live webcam detection:

```bash
python -m streamlit run streamlit_app.py
