from ultralytics import YOLO
import pandas as pd

# Load YOLO base model
model = YOLO("yolov8n.pt")   # n = nano (small/fast), try s/m for better accuracy later

# Train the model
model.train(
    data="C:/food-project/datasets/data.yaml",  # path to your dataset yaml
    epochs=50,         # number of training rounds
    imgsz=640,         # image size
    batch=16,          # batch size (adjust if low RAM)
    name="food_detector"
)
