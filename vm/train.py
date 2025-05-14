from ultralytics import YOLO
import os
from pathlib import Path
from data_generator import DataGenerator

image_per_run = 10
data_generator = DataGenerator()
data_generator.generate_yaml()
data_generator.generate_data(image_per_run)
data_generator.split_data()

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
DATASET_DIR = Path(dataset_dir)
DATA_YAML = DATASET_DIR / "data.yaml"

if not DATA_YAML.exists():
  print(f"Dataset YAML file not found at {DATA_YAML.resolve()}")
  exit(1)

# First run
model_path = os.path.join(base_dir, "yolo11m.pt")
model = YOLO(model = model_path)

results = model.train(
  data=DATA_YAML,
  epochs=2,
  imgsz=640,
)

results = model.val(
  data = DATA_YAML,
)

for i in range(5):
  data_generator.generate_data(image_per_run)
  data_generator.split_data()

  model = YOLO(model_path)

  results = model.train(
    data=DATA_YAML,
    epochs=1,
    imgsz=640,
  )

  results = model.val(
    data = DATA_YAML,
  )

model.export(format = "tflite")
