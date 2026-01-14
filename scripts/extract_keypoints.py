import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.pose_extractor import extract_pose_keypoints

DATASET_PATH = "dataset/RWF2000/val"
SAVE_PATH = "extracted_keypoints/val"

for label in ["Fight", "NonFight"]:
    os.makedirs(os.path.join(SAVE_PATH, label), exist_ok=True)
    video_dir = os.path.join(DATASET_PATH, label)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)

        keypoints = extract_pose_keypoints(video_path)

        if keypoints is not None:
            save_name = video_name.replace(".avi", ".npy")
            np.save(os.path.join(SAVE_PATH, label, save_name), keypoints)

        print(f"Saved: {label}/{video_name}")
