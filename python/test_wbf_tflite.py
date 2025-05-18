import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


models = [
  {"path": "../app/app/src/main/assets/best_env.tflite", "label": "../app/app/src/main/assets/labels.txt"},
  {"path": "../app/app/src/main/assets/best_clipped.tflite", "label": "../app/app/src/main/assets/labels.txt"},
]

interpreters = []
labels_list = []
input_details_list = []
output_details_list = []

for model in models:
  interpreter = tflite.Interpreter(model["path"])
  interpreter.allocate_tensors()
  interpreters.append(interpreter)

  label = model["label"]
  with open(label, "r", encoding="utf-8") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}
  labels_list.append(labels)

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  input_details_list.append(input_details)
  output_details_list.append(output_details)

# in_width, in_height = input_details["shape"][1:3]
# in_index = input_details["index"]
# in_scale, in_zero_point = input_details["quantization"]
# int8 = input_details["dtype"] == np.int8
# out_index = output_details["index"]
# out_scale, out_zero_point = output_details["quantization"]

np.random.seed(42)
color_palette = np.random.uniform(128, 255, size=(len(labels), 3))

# Configuration
conf_threshold = 0.8 # Lowered for debugging
iou_threshold = 0.7
image_path = "../assets/test_set/12.png"

def compute_iou(box1, box2):
  """Compute IoU between two boxes: [x1, y1, x2, y2]"""
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  union = area1 + area2 - inter_area
  return inter_area / union if union != 0 else 0

def weighted_fusion(boxes, scores):
  """Fuse overlapping boxes using score-weighted average"""
  fused = []
  used = [False] * len(boxes)

  for i, box1 in enumerate(boxes):
    if used[i]:
      continue
    group = [(box1, scores[i])]
    used[i] = True

    for j in range(i + 1, len(boxes)):
      if used[j]:
        continue
      iou = compute_iou(box1, boxes[j])
      if iou > iou_threshold:
        group.append((boxes[j], scores[j]))
        used[j] = True

    # Compute weighted average
    total_score = sum(score for _, score in group)
    x1 = sum(b[0] * s for b, s in group) / total_score
    y1 = sum(b[1] * s for b, s in group) / total_score
    x2 = sum(b[2] * s for b, s in group) / total_score
    y2 = sum(b[3] * s for b, s in group) / total_score
    fused.append(([x1, y1, x2, y2], total_score / len(group)))

  return fused

# Letterbox preprocessing 
def letterbox(img, new_shape=(640, 640)):
  shape = img.shape[:2]  # [height, width]
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
  if shape[::-1] != new_unpad:
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return img, (top / img.shape[0], left / img.shape[1])

# Draw detections 
def draw_detections(img, box, score, class_id):
  x1, y1, w, h = box
  color = color_palette[class_id]
  cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
  label = f"{labels[class_id]}: {score:.2f}"
  (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
  label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
  cv2.rectangle(img, (int(label_x), int(label_y - label_height)), (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
  cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def wbf(all_detections):
  final_results = []
  for cls in np.unique([det["class_id"] for det in all_detections]):
    cls_dets = [det for det in all_detections if det["class_id"] == cls]
    boxes = [det["box"] for det in cls_dets]
    scores = [det["score"] for det in cls_dets]
    fused = weighted_fusion(boxes, scores)
    final_results.extend([(box, score, cls) for box, score in fused])
  return final_results

all_detections = []
orig_img = cv2.imread(image_path)
orig_h, orig_w = orig_img.shape[:2]
for i, interpreter in enumerate(interpreters):
  input_details = input_details_list[i]
  output_details = output_details_list[i]
  in_width, in_height = input_details["shape"][1:3]
  in_scale, in_zero_point = input_details["quantization"]
  int8 = input_details["dtype"] == np.int8
  out_scale, out_zero_point = output_details["quantization"]
  labels = labels_list[i]

  # Preprocess image
  img, pad = letterbox(orig_img.copy(), new_shape=(in_width, in_height))
  img = img[..., ::-1][None]  # BGR to RGB, add batch dimension
  img = np.ascontiguousarray(img)
  input_data = img.astype(np.float32) / 255.0
  if int8:
    input_data = (input_data / in_scale + in_zero_point).astype(np.int8)

  # Run inference
  in_index = input_details["index"]
  out_index = output_details["index"]
  interpreter.set_tensor(in_index, input_data)
  interpreter.invoke()
  output = interpreter.get_tensor(out_index)
  if int8:
    output = (output.astype(np.float32) - out_zero_point) * out_scale

  # Postprocess output
  output[:, 0] -= pad[1]
  output[:, 1] -= pad[0]
  output[:, :4] *= max(orig_h, orig_w)
  output = output.transpose(0, 2, 1)
  output[..., 0] -= output[..., 2] / 2  # x center to top-left x
  output[..., 1] -= output[..., 3] / 2  # y center to top-left y

  for out in output:
    scores = out[:, 4:].max(-1)
    keep = scores > 0.8  # 你設定的conf threshold
    boxes = out[keep, :4]
    scores = scores[keep]
    class_ids = out[keep, 4:].argmax(-1)

    # 轉成x1,y1,x2,y2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    for cls in np.unique(class_ids):
      idx = class_ids == cls
      cls_boxes = boxes[idx]
      cls_scores = scores[idx]
      
      # 把此模型的結果暫存
      for box, score in zip(cls_boxes, cls_scores):
        all_detections.append({
          "box": box.tolist(),
          "score": float(score),
          "class_id": int(cls),
          "model_idx": i,
        })
        # print(f"Model {i}, Class {cls}, Box: {box}, Score: {score}")

final_result = wbf(all_detections)
print("Final results after WBF:")
for box, score, cls in final_result:
  print(f"Class {cls}, Box: {box}, Score: {score}")

# Debug with OpenCV
detected_img = orig_img.copy()
for box, score, cls in final_result:
  draw_detections(detected_img, box, score, cls)
  
cv2.imshow("Output", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
