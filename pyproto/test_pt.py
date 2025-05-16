from ultralytics import YOLO
from pathlib import Path

# Folders
base_dir = Path(__file__).resolve().parent
test_set_dir = base_dir / '../assets/test_set'
test_results_dir = base_dir / '../assets/test_result'

test_set_dir = test_set_dir.resolve()
test_results_dir = test_results_dir.resolve()

# Load a model
model = YOLO(base_dir / 'yolo11s.pt')

# Run batched inference on a list of images
test_set = []
for i in range(1, 8):
  test_set.append(test_set_dir / f"{i}.png")

results = model.predict(test_set, save=False, conf=0.5, imgsz=640)
