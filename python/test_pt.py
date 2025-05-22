from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# === 資料夾路徑 ===
base_dir = Path(__file__).resolve().parent
test_set_dir = (base_dir / '../assets/test_set').resolve()
test_results_dir = (base_dir / '../assets/test_result').resolve()
test_answer_dir = (base_dir / '../assets/test_answer').resolve()
model_path = (base_dir / '../assets/test_model/m_30000_0522.pt').resolve()

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
  height, width = img.shape[:2]
  results = model.predict(source=img, conf=0.4, iou=0.7, verbose=False)

  txt_lines = []

  # 繪製預測框
  for r in results:
    for box in r.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      cls_id = int(box.cls)
      conf = float(box.conf)
      color = color_palette[cls_id]
      label = f"{model.names[cls_id]} {conf:.2f}"

      # 計算 YOLO 格式座標（中心點＋寬高，相對比例）
      x_center = ((x1 + x2) / 2) / width
      y_center = ((y1 + y2) / 2) / height
      w = (x2 - x1) / width
      h = (y2 - y1) / height

      txt_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
      txt_lines.append(txt_line)

      print(f"檢測到：{label}, Confidence: {conf:.2f}")

      # 繪圖
      cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
      (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, 0)
      cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  # 儲存圖片
  output_path = test_results_dir / image_path.name
  cv2.imwrite(str(output_path), img)
  print(f"✅ 已輸出：{output_path.name}")

  # # 儲存 txt（同名，不同副檔名）
  # txt_output_path = test_answer_dir / (image_path.stem + ".txt")
  # with open(txt_output_path, "w") as f:
  #   f.write("\n".join(txt_lines))
  # print(f"📝 已輸出標註：{txt_output_path.name}")
