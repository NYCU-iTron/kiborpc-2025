from ultralytics import YOLO
from pathlib import Path

DATASET = Path("../../assets/dataset/data.yaml")

if not DATASET.exists():
    print(f"Dataset YAML file not found at {DATASET.resolve()}")
    exit(1)

# # start from scratch
# model = YOLO(model = "yolo11n.yaml")
# load pretrained weights
model = YOLO(model = "yolo11n.pt")

results = model.train(
    data = DATASET,
    epochs = 100
)
results = model.val(
    data = DATASET,
)

model.export(format = "tflite")