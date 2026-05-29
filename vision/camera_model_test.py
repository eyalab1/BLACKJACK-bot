import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.card_detector import CardDetector


CAMERA_INDEX = 2
MODEL_PATH = "models/cards_yolo.pt"

WINDOW_NAME = "Blackjack New Vision - Model Test"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Put your model here: models/cards_yolo.pt")
        return

    detector = CardDetector(
        model_path=MODEL_PATH,
        conf=0.35,
        iou=0.45,
        min_area=700,
        history_size=10,
        stable_count=6,
        duplicate_distance=55,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Camera not opened.")
        print("Try changing CAMERA_INDEX from 2 to 0 or 1.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Camera model test started.")
    print("Q = quit")
    print("R = reset stability history")
    print("S = print stable cards")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        detections = detector.detect_frame(frame)
        frame = detector.draw(frame, detections)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            break

        if key in [ord("r"), ord("R")]:
            detector.reset()
            print("History reset.")

        if key in [ord("s"), ord("S")]:
            stable_cards = detector.get_stable_cards()
            stable_values = detector.get_stable_blackjack_values()

            print("=" * 50)
            print("Stable cards:", stable_cards)
            print("Blackjack values:", stable_values)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()