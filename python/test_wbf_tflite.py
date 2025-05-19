import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pathlib import Path

class Detection:
  def __init__(self, box, score, class_id, class_name, model_name=None, model_weight=1.0):
    self.box = box
    self.score = score
    self.class_id = class_id
    self.class_name = class_name
    self.model_name = model_name
    self.model_weight = model_weight

class Interpreter:
  def __init__(self, model_name, model_weight=1.0):
    """
    Load TFLite model and allocate tensors.
    """
    # Get the path to the model file
    base_dir = Path(__file__).resolve().parent
    assets_dir = (base_dir / '../app/app/src/main/assets').resolve()
    model_path = (assets_dir / model_name).resolve()
    if not model_path.exists():
      raise FileNotFoundError(f"Model file not found: {model_path}")
    
    self.model_name = model_name
    self.model_weight = model_weight
    
    # Load the TFLite model
    self.interpreter = tflite.Interpreter(str(model_path))
    self.interpreter.allocate_tensors()

    # Store details
    self.input_details = self.interpreter.get_input_details()[0]
    self.output_details = self.interpreter.get_output_details()[0]

    # Load labels
    labels_path = (assets_dir / "labels_v2.txt").resolve()
    if not labels_path.exists():
      raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, "r", encoding="utf-8") as f:
      self.labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    self.conf_threshold = 0.4
  
  def detect(self, image):
    in_scale, in_zero_point = self.input_details["quantization"]
    out_scale, out_zero_point = self.output_details["quantization"]
    int8 = self.input_details["dtype"] == np.int8

    # Preprocess image
    img, pad = self.letterbox(image.copy())
    img = img[..., ::-1][None]  # BGR to RGB, add batch dimension
    img = np.ascontiguousarray(img)
    input_data = img.astype(np.float32) / 255.0
    if int8:
      input_data = (input_data / in_scale + in_zero_point).astype(np.int8)

    # Run inference
    in_index = self.input_details["index"]
    out_index = self.output_details["index"]
    self.interpreter.set_tensor(in_index, input_data)
    self.interpreter.invoke()
    output = self.interpreter.get_tensor(out_index)
    if int8:
      output = (output.astype(np.float32) - out_zero_point) * out_scale

    # Postprocess output
    orig_h, orig_w = image.shape[:2]
    output[:, 0] -= pad[1]
    output[:, 1] -= pad[0]
    output[:, :4] *= max(orig_h, orig_w)
    output = output.transpose(0, 2, 1)
    output[..., 0] -= output[..., 2] / 2  # x center to top-left x
    output[..., 1] -= output[..., 3] / 2  # y center to top-left y

    detections = []
    for out in output:
      scores = out[:, 4:].max(-1)
      keep = scores > self.conf_threshold

      boxes = out[keep, :4]
      scores = scores[keep]
      class_ids = out[keep, 4:].argmax(-1)

      # Transform to x1, y1, x2, y2
      boxes[:, 2] += boxes[:, 0]
      boxes[:, 3] += boxes[:, 1]

      for cls in np.unique(class_ids):
        idx = class_ids == cls
        cls_boxes = boxes[idx]
        cls_scores = scores[idx]
        
        for box, score in zip(cls_boxes, cls_scores):
          detection = Detection(box=box.tolist(),
                                score=float(score),
                                model_weight=self.model_weight,
                                class_id=int(cls),
                                class_name=self.labels[int(cls)],
                                model_name=self.model_name)
          detections.append(detection)

    return detections

  # ------------------------------ Tool functions ------------------------------ #
  def letterbox(self, image):
    # Shapes
    shape = image.shape[:2]
    new_shape = self.input_details["shape"][1:3]

    # Resize
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
      img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    pad = (top / img.shape[0], left / img.shape[1])

    return img, pad
  
# ------------------------------ WBF functions ------------------------------ #
def compute_iou(box1, box2):
  """
  Compute IoU between two boxes: [x1, y1, x2, y2]
  """
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  union = area1 + area2 - inter_area
  return inter_area / union if union != 0 else 0

def is_contained(inner, outer, threshold=0.9):
  xi, yi, wi, hi = inner
  xo, yo, wo, ho = outer

  xi2, yi2 = xi + wi, yi + hi
  xo2, yo2 = xo + wo, yo + ho

  inter_x1 = max(xi, xo)
  inter_y1 = max(yi, yo)
  inter_x2 = min(xi2, xo2)
  inter_y2 = min(yi2, yo2)

  inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
  inner_area = wi * hi

  return (inter_area / inner_area) > threshold

def wbf(detections, iou_threshold=0.7, conf_threshold=0.5):
  # Arrange detections by score in descending order
  detections.sort(key=lambda x: x.score, reverse=True)

  fused = []
  used = [False] * len(detections)
  for i in range(len(detections)):
    if used[i]:
      continue

    group = [detections[i]]
    used[i] = True

    for j in range(i + 1, len(detections)):
      if used[j]:
        continue

      iou = compute_iou(detections[i].box, detections[j].box)
      contained = is_contained(detections[j].box, detections[i].box)
      if iou > iou_threshold or contained:
        group.append(detections[j])
        used[j] = True

    # Compute box
    total_score = sum(detection.score * detection.model_weight for detection in group)
    avg_score = sum(detection.score for detection in group) / sum(detection.model_weight for detection in group)
    
    if total_score < conf_threshold:
      continue

    x1 = sum(detection.box[0] * detection.score for detection in group) / total_score
    y1 = sum(detection.box[1] * detection.score for detection in group) / total_score
    x2 = sum(detection.box[2] * detection.score for detection in group) / total_score
    y2 = sum(detection.box[3] * detection.score for detection in group) / total_score
    w, h = abs(x2 - x1), abs(y2 - y1)
    x1, y1= min(x1, x2), min(y1, y2)
    box = [x1, y1, w, h]

    # Compute class id and name
    group.sort(key=lambda x: x.score * x.model_weight, reverse=True)
    detection = Detection(box=box, score=avg_score,
                          class_id=group[0].class_id,
                          class_name=group[0].class_name,
                          model_name=group[0].model_name)
    fused.append(detection)
  
  # Remove too small boxes
  fused = [detection for detection in fused if detection.box[2] > 10 and detection.box[3] > 10]

  return fused

def draw_detections(img, box, score, class_id):
  x1, y1, w, h = box
  color = color_palette[class_id]
  cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1)
  label = f"{labels[class_id]}: {score:.2f}"
  (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
  label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
  # cv2.rectangle(img, (int(label_x), int(label_y - label_height)), (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
  # cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

np.random.seed(42)
color_palette = np.random.uniform(128, 255, size=(11, 3))

# Configuration
image_path = "../assets/test_set/7.png"

# Load labels
base_dir = Path(__file__).resolve().parent
assets_dir = (base_dir / '../app/app/src/main/assets').resolve()
labels_path = (assets_dir / "labels_v2.txt").resolve()
if not labels_path.exists():
  raise FileNotFoundError(f"Labels file not found: {labels_path}")
with open(labels_path, "r", encoding="utf-8") as f:
  labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Interpreter
interpreter_0516 = Interpreter("s_15000_0516.tflite")
interpreter_0519 = Interpreter("n_20000_0519.tflite")

# Load image
orig_img = cv2.imread(image_path)
if orig_img is None:
  raise FileNotFoundError(f"Image file not found: {image_path}")

# Detect
detections_0516 = interpreter_0516.detect(orig_img)
detections_0519 = interpreter_0519.detect(orig_img)

# Process detections
all_detections = []
all_detections.extend(detections_0516)
all_detections.extend(detections_0519)

final_detections = wbf(all_detections)

print("Final results after WBF:")
for detection in final_detections:
  print(f"Class {labels[int(detection.class_id)]}, Score: {detection.score}")

# Debug with OpenCV
detected_img = orig_img.copy()
for detection in final_detections:
  draw_detections(detected_img, detection.box, detection.score, detection.class_id)
  
cv2.imshow("Output", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
