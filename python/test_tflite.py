import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

model = "../app/app/src/main/assets/best_clipped.tflite"
label = "../app/app/src/main/assets/labels.txt"

# === Load model ===
interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_width, in_height = input_details["shape"][1:3]
in_index = input_details["index"]
in_scale, in_zero_point = input_details["quantization"]
int8 = input_details["dtype"] == np.int8
out_index = output_details["index"]
out_scale, out_zero_point = output_details["quantization"]

# Load labels
with open(label, "r", encoding="utf-8") as f:
  labels = {i: line.strip() for i, line in enumerate(f.readlines())}
np.random.seed(42)
color_palette = np.random.uniform(128, 255, size=(len(labels), 3))

# Configuration
conf_threshold = 0.8 # Lowered for debugging
iou_threshold = 0.45
image_path = "1.png"

# Letterbox preprocessing 
def letterbox(img, new_shape=(in_height, in_width)):
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

# Preprocess image
orig_img = cv2.imread(image_path)
orig_h, orig_w = orig_img.shape[:2]
img, pad = letterbox(orig_img)
print("Letterboxed image shape:", img.shape)
img = img[..., ::-1][None]  # BGR to RGB, add batch dimension
img = np.ascontiguousarray(img)
input_data = img.astype(np.float32) / 255.0
if int8:
  input_data = (input_data / in_scale + in_zero_point).astype(np.int8)

# Run inference
interpreter.set_tensor(in_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(out_index)
print("Model output shape:", output.shape)
if int8:
  output = (output.astype(np.float32) - out_zero_point) * out_scale

# Postprocess output
output[:, 0] -= pad[1]
output[:, 1] -= pad[0]
output[:, :4] *= max(orig_h, orig_w)
output = output.transpose(0, 2, 1)
output[..., 0] -= output[..., 2] / 2  # x center to top-left x
output[..., 1] -= output[..., 3] / 2  # y center to top-left y

# Apply NMS
for out in output:
  scores = out[:, 4:].max(-1)
  keep = scores > conf_threshold
  boxes = out[keep, :4]
  scores = scores[keep]
  class_ids = out[keep, 4:].argmax(-1)
  indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
  if isinstance(indices, tuple):
    indices = indices[0]
  indices = indices.flatten()
  print("Number of detections after NMS:", len(indices))
  for i in indices:
    draw_detections(orig_img, boxes[i], scores[i], class_ids[i])
    print(class_ids[i], scores[i])

# Debug with OpenCV
cv2.imshow("Output", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
