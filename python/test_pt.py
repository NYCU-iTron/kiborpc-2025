from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# === 資料夾路徑 ===
base_dir = Path(__file__).resolve().parent
test_set_dir = (base_dir / '../assets/test_set').resolve()
test_results_dir = (base_dir / '../assets/test_result').resolve()
model_path = (base_dir / '../assets/test_model/s_15000_0516.pt').resolve()

# 確保輸出資料夾存在
test_results_dir.mkdir(parents=True, exist_ok=True)

# === 載入模型 ===
model = YOLO(str(model_path))  # YOLO 模型

# 隨機顏色（每個 class 一個顏色）
color_palette = np.random.uniform(128, 255, size=(len(model.names), 3))

# === 處理所有圖片 ===
image_paths = sorted(test_set_dir.glob("*.[jpJP]*[npNP]*[gG]"))  # jpg, png
for image_path in image_paths:
  img = cv2.imread(str(image_path))
  if img is None:
    print(f"⚠️ 無法讀取：{image_path.name}")
    continue

  # 推論
  results = model.predict(source=img, conf=0.4, iou=0.45, verbose=False)

  # 繪製預測框
  for r in results:
    for box in r.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      cls_id = int(box.cls)
      conf = float(box.conf)
      color = color_palette[cls_id]
      label = f"{model.names[cls_id]} {conf:.2f}"

      # 繪圖
      cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
      (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
      cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  # 儲存圖片
  output_path = test_results_dir / image_path.name
  cv2.imwrite(str(output_path), img)
  print(f"✅ 已輸出：{output_path.name}")
