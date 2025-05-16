from ultralytics import YOLO

# Folders
test_set_dir = '../assets/test_set'
test_results_dir = '../assets/test_results'

# Load a model
model = YOLO('yolo11s.pt')

# Run batched inference on a list of images
test_set = []
for i in range(1, 6):
  test_set.append(f"{test_set_dir}/{i}.png")

results = model(test_set)

# Process results list
for i, result in enumerate(results):
  boxes = result.boxes
  masks = result.masks
  keypoints = result.keypoints
  probs = result.probs
  obb = result.obb
  result.show()
  result.save(filename=f"{i}.png")
