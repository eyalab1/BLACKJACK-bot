import cv2
import os
from datetime import datetime

PHONE_CAMERA_URL = "http://192.168.1.207:8080/video"

cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():
    print("Could not open phone camera.")
    exit()

os.makedirs("data/cards", exist_ok=True)

print("Camera started.")
print("Press c = capture card")
print("Press q or ESC = quit")
print("You can also close the window with X")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Could not read frame.")
        break

    frame = cv2.resize(frame, (900, 600))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    biggest_box = None
    biggest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 5000:
            x, y, w, h = cv2.boundingRect(contour)

            if area > biggest_area:
                biggest_area = area
                biggest_box = (x, y, w, h)

    if biggest_box is not None:
        x, y, w, h = biggest_box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Possible Card",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("BlackJack Vision", frame)

    # מאפשר לסגור עם X
    if cv2.getWindowProperty("BlackJack Vision", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(30) & 0xFF

    if key == ord("c"):
        if biggest_box is not None:
            x, y, w, h = biggest_box
            card = frame[y:y+h, x:x+w]

            filename = datetime.now().strftime(
                "data/cards/card_%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(filename, card)
            print(f"Saved card: {filename}")
        else:
            print("No card detected.")

    elif key == ord("q") or key == 27:
        print("Closing...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")