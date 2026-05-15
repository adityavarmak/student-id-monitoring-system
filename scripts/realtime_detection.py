import cv2
from ultralytics import YOLO

# ==========================================
# LOAD MODELS
# ==========================================

# Person detection model
person_model = YOLO("yolov8n.pt")

# Your trained ID model
id_model = YOLO(
    r"C:\Users\adith\Downloads\id_card detection.v1i.yolov8\runs\detect\train\weights\best.pt"
)

# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# FUNCTION TO CHECK POINT INSIDE BOX
# ==========================================

def inside(person_box, obj_box):

    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box

    # Object center
    cx = (ox1 + ox2) // 2
    cy = (oy1 + oy2) // 2

    return px1 < cx < px2 and py1 < cy < py2

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ==========================================
    # PERSON DETECTION
    # ==========================================

    person_results = person_model(
        frame,
        conf=0.5,
        verbose=False
    )[0]

    # ==========================================
    # STRAP + CARD DETECTION
    # ==========================================

    id_results = id_model(
        frame,
        conf=0.75,
        verbose=False
    )[0]

    persons = []
    straps = []
    cards = []

    # ==========================================
    # EXTRACT PERSONS
    # ==========================================

    for box in person_results.boxes:

        cls_id = int(box.cls[0])

        # COCO class 0 = person
        if cls_id == 0:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            persons.append((x1, y1, x2, y2))

    # ==========================================
    # EXTRACT STRAPS & CARDS
    # ==========================================

    for box in id_results.boxes:

        cls_id = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        current_box = (x1, y1, x2, y2)

        # ------------------------------
        # CARD
        # ------------------------------

        if cls_id == 0:

            cards.append(current_box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "CARD",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ------------------------------
        # STRAP
        # ------------------------------

        elif cls_id == 1:

            straps.append(current_box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "STRAP",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    # ==========================================
    # CHECK EACH PERSON
    # ==========================================

    for person in persons:

        px1, py1, px2, py2 = person

        strap_found = False
        card_found = False

        # ------------------------------
        # CHECK STRAPS INSIDE PERSON
        # ------------------------------

        for strap in straps:

            if inside(person, strap):

                strap_found = True
                break

        # ------------------------------
        # CHECK CARDS INSIDE PERSON
        # ------------------------------

        for card in cards:

            if inside(person, card):

                card_found = True
                break

        # ======================================
        # FINAL STATUS
        # ======================================

        if strap_found and card_found:

            color = (0, 255, 0)
            label = "ID VERIFIED"

        else:

            color = (0, 0, 255)
            label = "ID NOT VERIFIED"

        # ======================================
        # DRAW PERSON BOX
        # ======================================

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            color,
            3
        )

        cv2.putText(
            frame,
            label,
            (px1, py1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # ==========================================
    # SHOW WINDOW
    # ==========================================

    cv2.imshow("Student ID Monitoring System", frame)

    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()
cv2.destroyAllWindows()