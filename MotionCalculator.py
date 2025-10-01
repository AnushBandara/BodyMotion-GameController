import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.keyboard import Key, Controller

# Camera setup (reduced resolution for speed)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = 0  # For FPS calculation
upCount = 0
downCount = 0
leftCount = 0
rightCount = 0

# Mediapipe Pose (optimized for speed)
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=0, smooth_landmarks=True)
mpDraw = mp.solutions.drawing_utils

# Keyboard controller
keyboard = Controller()

# Marker line positions
topMarker = 190
bottomMarker = 250
rightMarker = 290
leftMarker = 640 - rightMarker

# Vertical jump state flags (Default : 1 / Middle box)
pStateY = 1
cStateY = 1

# Horizontal jump state flags (Default : 1 / Middle box)
pStateX = 1
cStateX = 1

upKey = Key.up
leftKey = Key.left
rightKey = Key.right
downKey = Key.down

# Toggle drawing for speed test
DRAW_LANDMARKS = True

while True:
    ret, img = cap.read()
    if not ret:
        continue

    # Marker lines
    cv2.line(img, (640, topMarker), (0, topMarker), (255, 255, 255), 2)  # top line - White
    cv2.line(img, (640, bottomMarker), (0, bottomMarker), (0, 0, 255), 2)  # top line - Red
    cv2.line(img, (leftMarker, 0), (leftMarker, 480), (0, 255, 0), 2)  # left line - Green
    cv2.line(img, (rightMarker, 0), (rightMarker, 480), (255, 0, 0), 2)  # right line - Blue

    # Convert to RGB (contiguous memory)
    imgRGB = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        if DRAW_LANDMARKS:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        h, w, c = img.shape
        rightElbow = results.pose_landmarks.landmark[12]
        leftElbow = results.pose_landmarks.landmark[11]
        rx, ry = int(rightElbow.x * w), int(rightElbow.y * h)
        lx, ly = int(leftElbow.x * w), int(leftElbow.y * h)
        cx, cy = (rx + lx) // 2, (ry + ly) // 2

        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        # Jump vertical detection logic
        if cy < topMarker:
            cStateY = 0
            if pStateY > cStateY:
                keyboard.press(upKey)
                print("Up ^^")
                keyboard.release(upKey)
                upCount += 1
                pStateY = cStateY

        elif cy > bottomMarker:
            cStateY = 2
            if pStateY < cStateY:
                keyboard.press(downKey)
                print("Down !!")
                keyboard.release(downKey)
                downCount += 1
                pStateY = cStateY

        else:
            cStateY, pStateY = 1, 1

        # Jump horizontal detection logic
        if cx < rightMarker:
            # Right box
            cStateX = 2

            if pStateX < cStateX:
                keyboard.press(rightKey)
                print("Right ->")
                keyboard.release(rightKey)
                rightCount += 1
                pStateX = cStateX

        elif cx > leftMarker:
            # Left box
            cStateX = 0

            if pStateX > cStateX:
                keyboard.press(leftKey)
                print("<- Left")
                leftCount += 1
                keyboard.release(leftKey)
                pStateX = cStateX

        else:
            # Middle box
            cStateX = 1

            if pStateX < cStateX:
                # Left box to middle
                keyboard.press(rightKey)
                print("Right ->")
                rightCount += 1
                keyboard.release(rightKey)
                pStateX = cStateX

            elif pStateX > cStateX:
                # Right box to middle
                keyboard.press(leftKey)
                print("<- Left")
                leftCount += 1
                keyboard.release(leftKey)
                pStateX = cStateX

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime

    # Mirror image
    img_mirrored = cv2.flip(img, 1)

    # Overlay text
    cv2.putText(img_mirrored, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_mirrored, f"Up : {upCount}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_mirrored, f"Down : {downCount}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_mirrored, f"Left : {leftCount}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_mirrored, f"Right : {rightCount}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Live Cam", img_mirrored)
    cv2.setWindowProperty("Live Cam", cv2.WND_PROP_TOPMOST, 1) # Set "Live Cam" window to stay on top all windows

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
