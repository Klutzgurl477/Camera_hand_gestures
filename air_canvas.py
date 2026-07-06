import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIDTH, HEIGHT = 1280, 720

#Color Palette
COLORS = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("Eraser", (0, 0, 0)),  # special: erases
]
SWATCH_W = WIDTH // len(COLORS)

current_color = COLORS[0][1]
current_color_name = COLORS[0][0]
brush_thickness = 8
eraser_thickness = 40
# The persistent drawing canvas (same size as frame), starts blank (black -> treated as transparent)
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

prev_x, prev_y = None, None
xp, yp = 0, 0

# For FPS calculation
prev_time = 0


def fingers_up(landmarks, handedness_label):
    """
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb: compare x-coordinates (depends on hand orientation)
    if handedness_label == "Right":
        fingers.append(landmarks[tips[0]].x < landmarks[pips[0]].x)
    else:
        fingers.append(landmarks[tips[0]].x > landmarks[pips[0]].x)

    # Other four fingers: tip above pip joint (lower y = higher on screen)
    for i in range(1, 5):
        fingers.append(landmarks[tips[i]].y < landmarks[pips[i]].y)

    return fingers


def draw_color_bar(frame):
    for i, (name, color) in enumerate(COLORS):
        x1 = i * SWATCH_W
        x2 = x1 + SWATCH_W
        swatch_color = color if name != "Eraser" else (50, 50, 50)
        cv2.rectangle(frame, (x1, 0), (x2, 70), swatch_color, -1)
        text_color = (255, 255, 255) if name != "Yellow" else (0, 0, 0)
        cv2.putText(frame, name, (x1 + 10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, text_color, 2, cv2.LINE_AA)
        if color == current_color and name == current_color_name:
            cv2.rectangle(frame, (x1, 0), (x2, 70), (255, 255, 255), 3)


print("Air Canvas starting... press 'q' to quit, 's' to save, 'c' to clear.")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # MediaPipe wants RGB, OpenCV gives you BGR — convert
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if (result.multi_hand_landmarks and result.multi_handedness):
        hand_landmarks = result.multi_hand_landmarks[0]
        handedness_label = result.multi_handedness[0].classification[0].label
        landmarks = hand_landmarks.landmark
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

        up = fingers_up(landmarks, handedness_label)
        # up = [thumb, index, middle, ring, pinky]

        # --- Selection mode: index + middle up, others down ---
        if up[1] and up[2] and not up[3] and not up[4]:
            xp, yp = 0, 0  # reset stroke
            cv2.circle(frame, (ix, iy), 12, current_color, cv2.FILLED)
            cv2.circle(frame, (mx, my), 12, current_color, cv2.FILLED)

            if iy < 70:  # over the color bar
                idx = ix // SWATCH_W
                if 0 <= idx < len(COLORS):
                    current_color_name, current_color = COLORS[idx]

        # --- Draw mode: only index up ---
        elif up[1] and not up[2] and not up[3] and not up[4]:
            cv2.circle(frame, (ix, iy), brush_thickness, current_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = ix, iy

            thickness = eraser_thickness if current_color_name == "Eraser" else brush_thickness
            draw_color = (0, 0, 0) if current_color_name == "Eraser" else current_color
            cv2.line(canvas, (xp, yp), (ix, iy), draw_color, thickness)
            xp, yp = ix, iy

        # --- Clear gesture: open palm (all 4 fingers up incl. thumb) ---
        elif all(up):
            canvas[:] = 0
            xp, yp = 0, 0
            cv2.putText(frame, "Canvas Cleared!", (WIDTH // 2 - 150, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    # Merge canvas onto frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    frame = cv2.bitwise_or(frame_bg, canvas)

    draw_color_bar(frame)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (WIDTH - 150, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Index=Draw | Index+Middle=Select | Open Palm=Clear | q=Quit s=Save",
                (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('s'):
        os.makedirs("drawings", exist_ok=True)
        filename = f"drawings/air_canvas_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved drawing to {filename}")

cap.release()
cv2.destroyAllWindows()
