import sys
import time
import cv2
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("./runs_OLD_DATASET/segment/train/weights/best.pt")
    results = model.predict(source='../Dataset/video_dataset/videos/test/Cholec80-Video51-009.mp4', show=True, save=True)