from ultralytics import YOLO
import os
from pathlib import Path

epoches = 120

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
DATASET_DIR = Path(dataset_dir)
DATA_YAML = DATASET_DIR / "data.yaml"

if not DATA_YAML.exists():
  print(f"Dataset YAML file not found at {DATA_YAML.resolve()}")
  exit(1)

model_path = os.path.join(base_dir, "yolo11n.pt")

model = YOLO(model_path)
results = model.train(
  data=DATA_YAML,
  epochs=epoches,
  imgsz=640,
  batch=-1,
  device=0,
  plots=True,
  patience=30,
)
model.save(model_path)
model.export(format="tflite")