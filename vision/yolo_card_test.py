import cv2
from ultralytics import YOLO
from collections import Counter, deque

PHONE_CAMERA_URL = "http://192.168.1.207:8080/video"
MODEL_PATH = "data/models/playing_cards.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():
    print("Could not open phone camera.")
    exit()

recent_cards = deque(maxlen=20)

MIN_CONFIDENCE = 0.15
MIN_VOTES = 5

stable_card = None
current_card = None
best_confidence = 0.0


def draw_panel(frame, stable_card, current_card, confidence, votes_text):
    h, w = frame.shape[:2]

    # Top dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(
        frame,
        "BLACKJACK BOT - CARD VISION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Current detection
    current_text = f"Current: {current_card} ({confidence:.2f})" if current_card else "Current: Searching..."
    cv2.putText(
        frame,
        current_text,
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Stable detection
    stable_text = f"Stable: {stable_card}" if stable_card else "Stable: Waiting for stable detection..."
    cv2.putText(
        frame,
        stable_text,
        (370, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0) if stable_card else (180, 180, 180),
        2
    )

    # Votes
    cv2.putText(
        frame,
        votes_text,
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    # Bottom instructions
    cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "Press Q/ESC to quit | Hold card steady for stable detection",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2
    )


print("YOLO card detection started.")
print("Press q or ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 400))

    results = model.predict(
        source=frame,
        conf=0.10,
        verbose=False
    )

    boxes = results[0].boxes

    current_card = None
    best_confidence = 0.0
    best_box = None

    if len(boxes) > 0:
        for box in boxes:
            confidence = float(box.conf[0])

            if confidence >= MIN_CONFIDENCE and confidence > best_confidence:
                class_id = int(box.cls[0])
                current_card = model.names[class_id]
                best_confidence = confidence
                best_box = box

    if current_card is not None:
        recent_cards.append(current_card)

    votes_text = f"Votes: 0/{len(recent_cards)}"

    if len(recent_cards) > 0:
        most_common_card, votes = Counter(recent_cards).most_common(1)[0]
        votes_text = f"Votes for {most_common_card}: {votes}/{len(recent_cards)}"

        if votes >= MIN_VOTES:
            stable_card = most_common_card

    # Clean frame instead of YOLO default plot
    display_frame = frame.copy()

    # Draw best detection manually
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"{current_card} {best_confidence:.2f}"
        cv2.rectangle(display_frame, (x1, y1 - 35), (x1 + 170, y1), (0, 255, 0), -1)
        cv2.putText(
            display_frame,
            label,
            (x1 + 8, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2
        )

    draw_panel(display_frame, stable_card, current_card, best_confidence, votes_text)

    cv2.imshow("BlackJack Bot Vision", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        print("Closing...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")