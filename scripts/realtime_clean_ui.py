import cv2

from scripts.image_detection import detect_image

# ==========================================
# WEBCAM FUNCTION
# ==========================================

def run_webcam():

    cap = cv2.VideoCapture(0)

    # Better FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ======================================
        # RUN AI DETECTION
        # ======================================

        result_frame, total, verified, not_verified = detect_image(frame)

        # ======================================
        # RETURN FRAME TO STREAMLIT
        # ======================================

        yield (
            result_frame,
            total,
            verified,
            not_verified
        )

    cap.release()