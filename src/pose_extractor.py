import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract_pose_keypoints(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    keypoints_seq = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()

        if not ret or frame is None:
            break   # ✅ IMPORTANT

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_seq.append(keypoints)

        frame_count += 1

    cap.release()

    if len(keypoints_seq) == 0:
        print("[WARNING] No pose detected in video")
        return None

    return np.array(keypoints_seq)
