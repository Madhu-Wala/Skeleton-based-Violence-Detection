import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mediapipe as mp
import cv2
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_extractor import extract_pose_keypoints
from src.predict import predict_action

# ---------------- MEDIAPIPE ---------------- #
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- SETTINGS ---------------- #
VIDEO_PATH  = r"C:\Violence_Detection\dataset\RWF2000\train\Fight\9ErNHIovPDI_0.avi" #Enter your video path here
MODEL_PATH  = "models/bilstm_model_165.h5"
MAX_FRAMES  = 50
# ------------------------------------------ #

# ---------- FULL VIDEO INFERENCE ---------- #
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("❌ Cannot open video:", VIDEO_PATH)

all_keypoints = []
# 1. Create a named window before the loop
cv2.namedWindow("Violence Detection - Live Skeleton", cv2.WINDOW_NORMAL)

# 2. Set the desired window size (e.g., 1280x720)
cv2.resizeWindow("Violence Detection - Live Skeleton", 1280, 720)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints PER FRAME
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    keypoints = np.zeros(165, dtype=np.float32)

    if results.pose_landmarks:
        data = []
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z, lm.visibility, 1.0])
        keypoints[:len(data)] = data[:165]

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    all_keypoints.append(keypoints)

    cv2.imshow("Violence Detection - Live Skeleton", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------- MODEL PREDICTION ---------- #
all_keypoints = np.array(all_keypoints)

if len(all_keypoints) == 0:
    print("❌ No keypoints extracted")
    exit()

label, confidence = predict_action(
    MODEL_PATH,
    all_keypoints,
    max_frames=MAX_FRAMES
)

print(f"✅ Prediction: {label}, Confidence: {confidence:.2f}")
