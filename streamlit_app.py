import streamlit as st
import cv2
import numpy as np

from scripts.image_detection import detect_image
from scripts.realtime_clean_ui import run_webcam

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Student ID Monitoring System",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    "<h1 style='font-size:48px;'>AI-Based Student ID Monitoring System</h1>",
    unsafe_allow_html=True
)

st.info(
    "This system verifies whether students are wearing valid ID cards using YOLOv8 object detection."
)
st.write(
    "Real-time student ID verification using YOLOv8."
)

# ==========================================
# MODEL METRICS
# ==========================================

st.sidebar.title("Model Evaluation")

st.sidebar.metric(
    "Precision",
    "91.55%"
)

st.sidebar.metric(
    "Recall",
    "92.86%"
)

st.sidebar.metric(
    "F1 Score",
    "92.2%"
)

st.sidebar.metric(
    "mAP50",
    "86.39%"
)

st.sidebar.metric(
    "mAP50-95",
    "71.62%"
)

# ==========================================
# MODE SELECTION
# ==========================================

mode = st.sidebar.selectbox(
    "Select Detection Mode",
    ["Image Detection", "Live Webcam Detection"]
)

# ==========================================
# IMAGE DETECTION
# ==========================================

if mode == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Convert uploaded image
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes, 1)

        # ======================================
        # RUN AI DETECTION
        # ======================================

        with st.spinner("Running AI Detection..."):

            result_frame, total, verified, not_verified = detect_image(frame)

        # ======================================
        # SHOW IMAGE
        # ======================================

        st.image(
            result_frame,
            channels="BGR",
            caption="Detection Result",
            use_container_width=True
        )

        # ======================================
        # STATS
        # ======================================

        st.subheader("Detection Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Students Detected",
            total
        )

        col2.metric(
            "ID Verified",
            verified
        )

        col3.metric(
            "ID Not Verified",
            not_verified
        )

# ==========================================
# LIVE WEBCAM DETECTION
# ==========================================

elif mode == "Live Webcam Detection":

    start = st.button("Start Webcam")

    stop = st.button("Stop Webcam")

    FRAME_WINDOW = st.image([])

    stats_placeholder = st.empty()

    if start:

        for (
            result_frame,
            total,
            verified,
            not_verified
        ) in run_webcam():

            # ==================================
            # SHOW FRAME
            # ==================================

            FRAME_WINDOW.image(
                result_frame,
                channels="BGR"
            )

            # ==================================
            # SHOW LIVE STATS
            # ==================================

            with stats_placeholder.container():

                st.subheader("Live Detection Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Students Detected",
                    total
                )

                col2.metric(
                    "ID Verified",
                    verified
                )

                col3.metric(
                    "ID Not Verified",
                    not_verified
                )

            # ==================================
            # STOP BUTTON
            # ==================================

            if stop:
                break
st.markdown("---")

st.caption(
    "Developed using YOLOv8, OpenCV, and Streamlit"
)