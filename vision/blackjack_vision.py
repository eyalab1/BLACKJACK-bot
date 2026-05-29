import cv2
import os
import sys
from collections import Counter, deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.card_detector import CardDetector, CardDetection


CAMERA_INDEX = 2
MODEL_PATH = "models/cards_yolo.pt"

WINDOW_NAME = "Blackjack Vision"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

HISTORY_SIZE = 12
STABLE_COUNT = 6

# Above this line = dealer.
# Below this line = robot/player.
SPLIT_Y_RATIO = 0.48

# Physical card contour filtering
MIN_CARD_AREA = 5000
MAX_CARD_AREA = 120000
CARD_PADDING = 18

# Full-frame fallback:
# Used when YOLO sees a card but OpenCV does not find a full physical rectangle.
FALLBACK_CONF = 0.40
FALLBACK_MIN_DISTANCE = 180

# Crop second-pass:
# Used when OpenCV finds a physical card rectangle,
# but YOLO did not detect a card inside it in the full frame.
CROP_CONF = 0.12
CROP_MIN_AREA = 80
CROP_SCALE = 2.0

# Final duplicate cleaning.
SAME_PHYSICAL_CARD_DISTANCE = 95


class BlackjackVision:
    def __init__(self):
        self.detector = CardDetector(
            model_path=MODEL_PATH,
            conf=0.20,
            iou=0.35,
            min_area=250,
            history_size=HISTORY_SIZE,
            stable_count=STABLE_COUNT,
            duplicate_distance=80,
        )

        self.split_ratio = SPLIT_Y_RATIO

        self.dealer_history = deque(maxlen=HISTORY_SIZE)
        self.robot_history = deque(maxlen=HISTORY_SIZE)

        self.current_frame = None
        self.show_debug = False

    # ==========================================================
    # Physical card detection
    # ==========================================================

    def find_physical_cards(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, erode_kernel, iterations=1)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        card_boxes = []
        h, w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < MIN_CARD_AREA or area > MAX_CARD_AREA:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            if cw <= 0 or ch <= 0:
                continue

            ratio = max(cw, ch) / min(cw, ch)

            if ratio < 1.10 or ratio > 2.35:
                continue

            x1 = max(0, x - CARD_PADDING)
            y1 = max(0, y - CARD_PADDING)
            x2 = min(w - 1, x + cw + CARD_PADDING)
            y2 = min(h - 1, y + ch + CARD_PADDING)

            card_boxes.append((x1, y1, x2, y2))

        card_boxes = self.merge_overlapping_card_boxes(card_boxes)

        return card_boxes

    def merge_overlapping_card_boxes(self, boxes):
        merged = []

        for box in boxes:
            added = False

            for i, kept in enumerate(merged):
                if self.iou(box, kept) > 0.65:
                    merged[i] = self.union_box(box, kept)
                    added = True
                    break

            if not added:
                merged.append(box)

        return merged

    def union_box(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        return (
            min(ax1, bx1),
            min(ay1, by1),
            max(ax2, bx2),
            max(ay2, by2),
        )

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0

        return inter / union

    def point_inside_box(self, point, box):
        px, py = point
        x1, y1, x2, y2 = box

        return x1 <= px <= x2 and y1 <= py <= y2

    def detection_near_any_physical_card(self, det, physical_card_boxes):
        cx, cy = det.center

        for box in physical_card_boxes:
            x1, y1, x2, y2 = box

            expanded = (
                x1 - FALLBACK_MIN_DISTANCE,
                y1 - FALLBACK_MIN_DISTANCE,
                x2 + FALLBACK_MIN_DISTANCE,
                y2 + FALLBACK_MIN_DISTANCE,
            )

            if self.point_inside_box((cx, cy), expanded):
                return True

        return False

    # ==========================================================
    # Crop second-pass detection
    # ==========================================================

    def detect_from_card_crop(self, frame, card_box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = card_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]

        if crop_h < 30 or crop_w < 30:
            return []

        enlarged = cv2.resize(
            crop,
            None,
            fx=CROP_SCALE,
            fy=CROP_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )

        crop_detections = self.detector.detect_image_no_history(
            image=enlarged,
            conf_override=CROP_CONF,
            min_area_override=CROP_MIN_AREA,
            x_offset=0,
            y_offset=0,
            max_det=10,
        )

        corrected = []

        for det in crop_detections:
            bx1, by1, bx2, by2 = det.box

            original_box = (
                int(x1 + bx1 / CROP_SCALE),
                int(y1 + by1 / CROP_SCALE),
                int(x1 + bx2 / CROP_SCALE),
                int(y1 + by2 / CROP_SCALE),
            )

            corrected.append(
                CardDetection(
                    card=det.card,
                    rank=det.rank,
                    suit=det.suit,
                    confidence=det.confidence,
                    box=original_box,
                )
            )

        return corrected

    # ==========================================================
    # Group detections
    # ==========================================================

    def group_detections_by_physical_card(self, detections, physical_card_boxes):
        final_detections = []
        used_detection_ids = set()

        frame = self.current_frame

        for card_box in physical_card_boxes:
            candidates = []

            for idx, det in enumerate(detections):
                cx, cy = det.center

                if self.point_inside_box((cx, cy), card_box):
                    candidates.append((idx, det))

            if candidates:
                best_idx, best_det = max(
                    candidates,
                    key=lambda item: item[1].confidence
                )

                final_detections.append(best_det)

                for idx, _ in candidates:
                    used_detection_ids.add(idx)

            else:
                if frame is not None:
                    crop_candidates = self.detect_from_card_crop(frame, card_box)

                    if crop_candidates:
                        best_crop_det = max(
                            crop_candidates,
                            key=lambda d: d.confidence
                        )

                        final_detections.append(best_crop_det)

        for idx, det in enumerate(detections):
            if idx in used_detection_ids:
                continue

            if det.confidence < FALLBACK_CONF:
                continue

            if self.detection_near_any_physical_card(det, physical_card_boxes):
                continue

            too_close_to_final = False

            for kept in final_detections:
                dx = det.center[0] - kept.center[0]
                dy = det.center[1] - kept.center[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < FALLBACK_MIN_DISTANCE:
                    too_close_to_final = True
                    break

            if not too_close_to_final:
                final_detections.append(det)

        final_detections = self.remove_same_position_duplicates(final_detections)

        return final_detections

    def remove_same_position_duplicates(self, detections):
        clean = []

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True
        )

        for det in detections:
            duplicate = False

            for kept in clean:
                dx = det.center[0] - kept.center[0]
                dy = det.center[1] - kept.center[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < SAME_PHYSICAL_CARD_DISTANCE:
                    duplicate = True
                    break

                if det.card == kept.card and dist < 130:
                    duplicate = True
                    break

            if not duplicate:
                clean.append(det)

        return clean

    # ==========================================================
    # Dealer / robot split
    # ==========================================================

    def split_detections(self, frame, detections):
        h, _ = frame.shape[:2]
        split_y = int(h * self.split_ratio)

        dealer_cards = []
        robot_cards = []

        for det in detections:
            _, cy = det.center

            if cy < split_y:
                dealer_cards.append(det.card)
            else:
                robot_cards.append(det.card)

        dealer_cards = sorted(list(set(dealer_cards)))
        robot_cards = sorted(list(set(robot_cards)))

        self.dealer_history.append(dealer_cards)
        self.robot_history.append(robot_cards)

        return dealer_cards, robot_cards

    def get_stable_from_history(self, history):
        counts = Counter()

        for cards_in_frame in history:
            for card in set(cards_in_frame):
                counts[card] += 1

        stable = []

        for card, count in counts.items():
            if count >= STABLE_COUNT:
                stable.append(card)

        return sorted(stable)

    def to_blackjack_values(self, cards):
        return self.detector.cards_to_blackjack_values(cards)

    def get_stable_state(self):
        stable_dealer_raw = self.get_stable_from_history(self.dealer_history)
        stable_robot_raw = self.get_stable_from_history(self.robot_history)

        return {
            "dealer_raw_cards": stable_dealer_raw,
            "robot_raw_cards": stable_robot_raw,
            "dealer_cards": self.to_blackjack_values(stable_dealer_raw),
            "robot_cards": self.to_blackjack_values(stable_robot_raw),
        }

    def reset(self):
        self.detector.reset()
        self.dealer_history.clear()
        self.robot_history.clear()

    def move_split_up(self):
        self.split_ratio = max(0.20, self.split_ratio - 0.03)

    def move_split_down(self):
        self.split_ratio = min(0.80, self.split_ratio + 0.03)

    def toggle_debug(self):
        self.show_debug = not self.show_debug
        print(f"Debug view: {self.show_debug}")

    # ==========================================================
    # Drawing helpers
    # ==========================================================

    def draw_panel(self, frame, x1, y1, x2, y2, alpha=0.72):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_text(self, frame, text, x, y, color, scale=0.58, thickness=1):
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_label_box(self, frame, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        tw, th = text_size

        cv2.rectangle(
            frame,
            (x - 3, y - th - 6),
            (x + tw + 5, y + 4),
            (10, 10, 10),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_split_line(self, frame, y):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.line(
            overlay,
            (0, y),
            (w, y),
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def draw_physical_cards(self, frame, physical_card_boxes):
        if not self.show_debug:
            return

        for box in physical_card_boxes:
            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (180, 0, 180),
                1,
                lineType=cv2.LINE_AA,
            )

    def draw_final_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det.box
            label = f"{det.card} {det.confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 210, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            self.draw_label_box(
                frame,
                label,
                x1,
                max(23, y1 - 8),
                (0, 230, 255),
            )

    def draw_interface(
        self,
        frame,
        raw_detections,
        final_detections,
        physical_card_boxes,
        current_dealer,
        current_robot,
    ):
        h, w = frame.shape[:2]
        split_y = int(h * self.split_ratio)
        stable_state = self.get_stable_state()

        # Panels
        self.draw_panel(frame, 0, 0, w, 82, alpha=0.76)
        self.draw_panel(frame, 0, h - 34, w, h, alpha=0.76)

        # Split line
        self.draw_split_line(frame, split_y)

        # Debug physical boxes only when D is on
        self.draw_physical_cards(frame, physical_card_boxes)

        # Final YOLO detections
        self.draw_final_detections(frame, final_detections)

        # Area labels
        self.draw_label_box(
            frame,
            "DEALER",
            14,
            max(112, split_y - 14),
            (245, 245, 245),
        )

        self.draw_label_box(
            frame,
            "ROBOT / PLAYER",
            14,
            min(h - 52, split_y + 34),
            (0, 245, 90),
        )

        # Header texts
        self.draw_text(
            frame,
            f"YOLO raw: {len(raw_detections)}   Final cards: {len(final_detections)}   Debug: {'ON' if self.show_debug else 'OFF'}",
            14,
            24,
            (0, 220, 255),
            scale=0.54,
        )

        self.draw_text(
            frame,
            f"Dealer now: {current_dealer}   |   stable: {stable_state['dealer_raw_cards']} -> {stable_state['dealer_cards']}",
            14,
            50,
            (235, 235, 235),
            scale=0.52,
        )

        self.draw_text(
            frame,
            f"Robot now:  {current_robot}   |   stable: {stable_state['robot_raw_cards']} -> {stable_state['robot_cards']}",
            14,
            75,
            (0, 245, 90),
            scale=0.52,
        )

        # Bottom help
        self.draw_text(
            frame,
            "S print stable   R reset   W split up   X split down   D debug   Q quit",
            14,
            h - 11,
            (225, 225, 225),
            scale=0.52,
        )

        return frame


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Put model here: models/cards_yolo.pt")
        return

    vision = BlackjackVision()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Camera not opened.")
        print("Try changing CAMERA_INDEX to 0, 1, or 2.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Requested camera resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Actual camera resolution:    {actual_w}x{actual_h}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Blackjack Vision started.")
    print("S = print stable state")
    print("R = reset")
    print("W = move split line up")
    print("X = move split line down")
    print("D = toggle debug boxes")
    print("Q = quit")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        vision.current_frame = frame.copy()

        raw_detections = vision.detector.detect_frame(frame)

        physical_card_boxes = vision.find_physical_cards(frame)

        final_detections = vision.group_detections_by_physical_card(
            raw_detections,
            physical_card_boxes,
        )

        current_dealer, current_robot = vision.split_detections(
            frame,
            final_detections,
        )

        frame = vision.draw_interface(
            frame=frame,
            raw_detections=raw_detections,
            final_detections=final_detections,
            physical_card_boxes=physical_card_boxes,
            current_dealer=current_dealer,
            current_robot=current_robot,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            break

        if key in [ord("r"), ord("R")]:
            vision.reset()
            print("History reset.")

        if key in [ord("w"), ord("W")]:
            vision.move_split_up()
            print("Split line moved up.")

        if key in [ord("x"), ord("X")]:
            vision.move_split_down()
            print("Split line moved down.")

        if key in [ord("d"), ord("D")]:
            vision.toggle_debug()

        if key in [ord("s"), ord("S")]:
            state = vision.get_stable_state()
            print("=" * 60)
            print("VISION STATE")
            print(state)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()