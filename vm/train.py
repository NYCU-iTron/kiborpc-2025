from ultralytics import YOLO
import os
from pathlib import Path
import torch

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
device = 0 if torch.cuda.is_available() else "cpu"
results = model.train(
  data=DATA_YAML,
  epochs=epoches,
  imgsz=320,
  batch=-1,
  device=device,
  plots=True,
  patience=30,
)
model.save(model_path)
model.export(format="tflite")