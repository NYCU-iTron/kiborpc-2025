from ultralytics import YOLO
import os
from pathlib import Path
from data_generator import DataGenerator

total_images = 20000
epoches = 100

data_generator = DataGenerator()
data_generator.generate_yaml()
data_generator.generate_data(total_images)
data_generator.split_data()

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
DATASET_DIR = Path(dataset_dir)
DATA_YAML = DATASET_DIR / "data.yaml"

if not DATA_YAML.exists():
  print(f"Dataset YAML file not found at {DATA_YAML.resolve()}")
  exit(1)

model_path = os.path.join(base_dir, "yolo11m.pt")

model = YOLO()
results = model.train(
  data=DATA_YAML,
  epochs=epoches,
  imgsz=640,
  batch=-1,
  device=0,
  plots=True,
)
model.save(model_path)