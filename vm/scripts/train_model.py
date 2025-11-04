from ultralytics import YOLO
from pathlib import Path
import torch

epochs = 2

base_dir = Path(__file__).resolve().parent.parent
dataset_dir = base_dir / "dataset"
data_yaml = dataset_dir / "data.yaml"

if not data_yaml.exists():
    raise FileNotFoundError(f"Dataset YAML file not found at {data_yaml.as_posix()}. Please generate the dataset first.")

model_path = base_dir / "yolo11n.pt"

model = YOLO(model_path)
device = 0 if torch.cuda.is_available() else "cpu"
results = model.train(
    data=data_yaml.as_posix(),
    epochs=epochs,
    imgsz=320,
    batch=-1,
    device=device,
    plots=True,
    patience=100,
)
model.save(model_path)
model.export(format="tflite")
