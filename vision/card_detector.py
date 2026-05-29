from dataclasses import dataclass
from collections import Counter, deque
from ultralytics import YOLO
import cv2
import math


@dataclass
class CardDetection:
    card: str
    rank: str
    suit: str
    confidence: float
    box: tuple

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class CardDetector:
    """
    YOLO card detector.

    This file only detects cards from an image/frame.
    It does not decide dealer/player.
    It does not decide blackjack actions.

    blackjack_vision.py handles:
    - physical card grouping
    - crop second-pass detection
    - dealer/robot split
    - stable state
    """

    def __init__(
        self,
        model_path="models/cards_yolo.pt",
        conf=0.20,
        iou=0.35,
        min_area=250,
        history_size=12,
        stable_count=6,
        duplicate_distance=80,
    ):
        self.model = YOLO(model_path)

        self.conf = conf
        self.iou = iou
        self.min_area = min_area

        self.history_size = history_size
        self.stable_count = stable_count
        self.duplicate_distance = duplicate_distance

        self.history = deque(maxlen=history_size)

        print("Model loaded successfully")
        print("Model classes:")
        print(self.model.names)

    # ==========================================================
    # Card name normalization
    # ==========================================================

    def normalize_card_name(self, raw_name):
        """
        Converts model class names into a clean card format.

        Supported examples:
        AS
        AH
        AD
        AC
        10S
        KD
        ace of spades
        king_of_hearts
        10 diamonds
        """

        if raw_name is None:
            return None

        name = str(raw_name).strip().upper()

        name = name.replace(" ", "")
        name = name.replace("_", "")
        name = name.replace("-", "")
        name = name.replace("OF", "")

        replacements = {
            "ACE": "A",
            "KING": "K",
            "QUEEN": "Q",
            "JACK": "J",
            "SPADES": "S",
            "SPADE": "S",
            "HEARTS": "H",
            "HEART": "H",
            "DIAMONDS": "D",
            "DIAMOND": "D",
            "CLUBS": "C",
            "CLUB": "C",
        }

        for old, new in replacements.items():
            name = name.replace(old, new)

        valid_suits = {"S", "H", "D", "C"}
        valid_ranks = {
            "A", "K", "Q", "J",
            "10", "9", "8", "7", "6", "5", "4", "3", "2"
        }

        if len(name) < 2:
            return None

        suit = name[-1]
        rank = name[:-1]

        if suit not in valid_suits:
            return None

        if rank not in valid_ranks:
            return None

        return rank + suit

    def split_card(self, card):
        rank = card[:-1]
        suit = card[-1]
        return rank, suit

    def to_blackjack_value(self, card):
        """
        Converts full card name to blackjack value.

        AS -> A
        KH -> 10
        9D -> 9
        """

        if card is None:
            return None

        rank = card[:-1]

        if rank in {"J", "Q", "K"}:
            return "10"

        return rank

    def cards_to_blackjack_values(self, cards):
        values = []

        for card in cards:
            value = self.to_blackjack_value(card)

            if value is not None:
                values.append(value)

        return values

    # ==========================================================
    # Geometry helpers
    # ==========================================================

    def box_area(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def center_distance(self, box_a, box_b):
        ax, ay = self.box_center(box_a)
        bx, by = self.box_center(box_b)

        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def iou_score(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)

        intersection = inter_w * inter_h

        area_a = self.box_area(box_a)
        area_b = self.box_area(box_b)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0

        return intersection / union

    def box_size_similarity(self, box_a, box_b):
        area_a = self.box_area(box_a)
        area_b = self.box_area(box_b)

        if area_a <= 0 or area_b <= 0:
            return 0

        small = min(area_a, area_b)
        large = max(area_a, area_b)

        return small / large

    # ==========================================================
    # YOLO detection
    # ==========================================================

    def detect_frame(self, frame):
        """
        Runs YOLO on the full frame.

        This returns raw YOLO detections after basic filtering only.
        Final duplicate/grouping logic is done in blackjack_vision.py.
        """

        detections = self.detect_image_no_history(
            image=frame,
            conf_override=self.conf,
            min_area_override=self.min_area,
            x_offset=0,
            y_offset=0,
            max_det=52,
        )

        self.history.append([d.card for d in detections])

        return detections

    def detect_image_no_history(
        self,
        image,
        conf_override=None,
        min_area_override=None,
        x_offset=0,
        y_offset=0,
        max_det=10,
    ):
        """
        Runs YOLO on an image/crop without updating history.

        Used for:
        - full frame detection
        - second-pass detection on cropped physical cards

        x_offset/y_offset convert crop coordinates back to full-frame coordinates.
        """

        conf = self.conf if conf_override is None else conf_override
        min_area = self.min_area if min_area_override is None else min_area_override

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.iou,
            verbose=False,
            max_det=max_det,
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_name = self.model.names[cls_id]

            card = self.normalize_card_name(raw_name)

            if card is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bbox = (
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
            )

            if self.box_area(bbox) < min_area:
                continue

            rank, suit = self.split_card(card)

            detections.append(
                CardDetection(
                    card=card,
                    rank=rank,
                    suit=suit,
                    confidence=confidence,
                    box=bbox,
                )
            )

        return detections

    # ==========================================================
    # Optional duplicate cleaning
    # ==========================================================

    def remove_duplicates(self, detections):
        """
        Optional duplicate cleaner.

        Main duplicate fix is in blackjack_vision.py:
        group by physical card rectangle and choose highest-confidence detection.
        """

        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True
        )

        clean = []

        for det in detections:
            duplicate = False

            for kept in clean:
                distance = self.center_distance(det.box, kept.box)
                overlap = self.iou_score(det.box, kept.box)
                size_similarity = self.box_size_similarity(det.box, kept.box)

                same_card = det.card == kept.card

                if same_card and (distance < self.duplicate_distance or overlap > 0.20):
                    duplicate = True
                    break

                if distance < self.duplicate_distance and size_similarity > 0.45:
                    duplicate = True
                    break

                if overlap > 0.35:
                    duplicate = True
                    break

            if not duplicate:
                clean.append(det)

        return clean

    # ==========================================================
    # Stability helpers
    # ==========================================================

    def get_stable_cards(self):
        counts = Counter()

        for cards_in_frame in self.history:
            for card in set(cards_in_frame):
                counts[card] += 1

        stable_cards = []

        for card, count in counts.items():
            if count >= self.stable_count:
                stable_cards.append(card)

        return sorted(stable_cards)

    def get_stable_blackjack_values(self):
        stable_cards = self.get_stable_cards()
        return self.cards_to_blackjack_values(stable_cards)

    def reset(self):
        self.history.clear()

    # ==========================================================
    # Drawing helpers
    # ==========================================================

    def draw_detections(self, frame, detections, color=(0, 220, 255), title=None):
        for det in detections:
            x1, y1, x2, y2 = det.box

            label = f"{det.card} {det.confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        if title:
            cv2.putText(
                frame,
                title,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        return frame

    def draw(self, frame, detections):
        self.draw_detections(frame, detections)

        stable_cards = self.get_stable_cards()
        stable_values = self.get_stable_blackjack_values()

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (20, 20, 20), -1)

        cv2.putText(
            frame,
            f"Detected now: {[d.card for d in detections]}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Stable cards: {stable_cards}",
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Blackjack values: {stable_values}",
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        return frame