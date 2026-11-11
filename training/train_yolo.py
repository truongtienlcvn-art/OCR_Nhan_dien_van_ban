# -*- coding: utf-8 -*-
"""
Script huấn luyện YOLOv8 để phát hiện biển số xe.
Chạy: python training/train_yolo.py
Sau khi chạy xong, weight tốt nhất nằm ở:
    training/runs/detect/train/weights/best.pt
Copy file đó vào ai_module/models/yolov8_plate.pt để hệ thống tự dùng.
"""
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset_detect", "data.yaml")

if __name__ == "__main__":
    # yolov8n = bản "nano", nhẹ nhất, phù hợp máy không có GPU rời (chạy CPU vẫn được)
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",       # đổi thành 0 nếu máy có GPU NVIDIA + CUDA
        project=os.path.join(BASE_DIR, "runs"),
        name="train",
        patience=15,        # dừng sớm nếu 15 epoch liền không cải thiện
    )

    print("\nHUAN LUYEN XONG!")
    print("Weight tot nhat nam o: training/runs/detect/train/weights/best.pt")