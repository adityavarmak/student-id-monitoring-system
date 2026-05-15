import cv2
from ultralytics import YOLO

# ==========================================
# LOAD MODELS
# ==========================================

person_model = YOLO(
    "models/yolov8n.pt"
)

id_model = YOLO(
    "models/best.pt"
)

# ==========================================
# FUNCTION
# ==========================================

def inside(person_box, obj_box):

    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box

    # Object center point
    cx = (ox1 + ox2) // 2
    cy = (oy1 + oy2) // 2

    return px1 < cx < px2 and py1 < cy < py2

# ==========================================
# MAIN DETECTION FUNCTION
# ==========================================

def detect_image(frame):

    # ======================================
    # PERSON DETECTION
    # ======================================

    person_results = person_model(
        frame,
        conf=0.5,
        verbose=False
    )[0]

    # ======================================
    # STRAP + CARD DETECTION
    # ======================================

    id_results = id_model(
        frame,
        conf=0.15,
        imgsz=800,
        verbose=False
    )[0]

    persons = []
    straps = []
    cards = []

    # ======================================
    # EXTRACT PERSONS
    # ======================================

    for box in person_results.boxes:

        cls_id = int(box.cls[0])

        # COCO class 0 = person
        if cls_id == 0:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            persons.append((x1, y1, x2, y2))

    # ======================================
    # EXTRACT STRAPS & CARDS
    # ======================================

    for box in id_results.boxes:

        cls_id = int(box.cls[0])

        class_name = id_model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        current_box = (x1, y1, x2, y2)

        # CARD
        if class_name == "Card":

            cards.append(current_box)

        # STRAP
        elif class_name == "Id strap":

            straps.append(current_box)

    verified_count = 0
    not_verified_count = 0

    # ======================================
    # CHECK EACH PERSON
    # ======================================

    for person in persons:

        px1, py1, px2, py2 = person

        strap_found = False
        card_found = False

        # CHECK STRAPS
        for strap in straps:

            if inside(person, strap):

                strap_found = True
                break

        # CHECK CARDS
        for card in cards:

            if inside(person, card):

                card_found = True
                break

        # ==================================
        # FINAL STATUS
        # ==================================

        if strap_found and card_found:

            color = (0, 255, 0)
            label = "ID VERIFIED"

            verified_count += 1

        else:

            color = (0, 0, 255)
            label = "ID NOT VERIFIED"

            not_verified_count += 1

        # ==================================
        # DRAW PERSON BOX
        # ==================================

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

    # ======================================
    # RETURN RESULTS
    # ======================================

    return (
        frame,
        len(persons),
        verified_count,
        not_verified_count
    )