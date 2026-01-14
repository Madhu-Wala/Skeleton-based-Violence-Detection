import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tensorflow as tf

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("MediaPipe OK")
print("TensorFlow:", tf.__version__)

YOLO("yolov8n.pt")
print("YOLO OK")
