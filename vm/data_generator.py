import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import random

class DataGenerator:
  def __init__(self):
    self.output_size = (640, 640)

    # Set the base directory to the directory of this script
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {self.base_dir}")
    
    # Check if the input directory exists
    self.input_dir = os.path.join(self.base_dir, 'item_images')
    if not os.path.exists(self.input_dir):
      print(f"{self.input_dir} does not exist. Please check the path.")

    # Create directories for images and labels
    self.dataset_dir = os.path.join(self.base_dir, 'dataset')
    self.image_dir = os.path.join(self.dataset_dir, 'images')
    self.label_dir = os.path.join(self.dataset_dir, 'labels')
    self.debug_dir = os.path.join(self.dataset_dir, 'debug')

    os.makedirs(self.dataset_dir, exist_ok=True)
    os.makedirs(self.image_dir, exist_ok=True)
    os.makedirs(self.label_dir, exist_ok=True)
    os.makedirs(self.debug_dir, exist_ok=True)

    if not os.path.exists(f'{self.base_dir}/dataset/classes.txt'):
      self.class_to_id = {
        'crystal': 0,
        'diamond': 1,
        'emerald': 2,
        'coin': 3,
        'compass': 4,
        'coral': 5,
        'fossil': 6,
        'key': 7,
        'letter': 8,
        'shell': 9,
        'treasure_box': 10
      }

      # 生成classes.txt，使用數字ID和對應的class_name
      with open(f'{self.base_dir}/dataset/classes.txt', 'w') as f:
        for class_name, class_id in self.class_to_id.items():
          f.write(f'{class_name}\n')
    else:
      with open(f'{self.base_dir}/dataset/classes.txt', 'r') as f:
        lines = f.readlines()
        self.class_to_id = {line.strip(): i for i, line in enumerate(lines)}
        print(f"Loaded classes from {self.base_dir}/dataset/classes.txt")

    self.items = self.load_items()

  def generate_data(self, total_image_count):
    count = 0

    for i in range(1, total_image_count+1):
      r = random.random()

      if r < 0.2:
        max_overlap = 0.7
      elif r > 0.2 and r < 0.5:
        max_overlap = 0.5
      else:
        max_overlap = 0.3

      image, annotations = self.generate_image(max_overlap=max_overlap)

      # 保存圖片
      cv2.imwrite(f'{self.image_dir}/image_{count + i:04d}.png', image)

      # 保存標註
      with open(f'{self.label_dir}/image_{count + i:04d}.txt', 'w') as f:
        f.write('\n'.join(annotations))

  def split_data(self):
    DATASET_DIR = Path(self.dataset_dir)
    IMAGES_DIR = DATASET_DIR / "images"

    # read images
    image_files = list(IMAGES_DIR.glob("*.*"))
    random.shuffle(image_files)

    split_idx = int(0.95 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # create folders
    for subset in ['train', 'val']:
      (DATASET_DIR / 'images' / subset).mkdir(parents=True, exist_ok=True)
      (DATASET_DIR / 'labels' / subset).mkdir(parents=True, exist_ok=True)

    self.move_files(train_files, 'train')
    self.move_files(val_files, 'val')

  def generate_yaml(self):
    DATASET_DIR = Path(self.dataset_dir)
    CLASSES_TXT = DATASET_DIR / "classes.txt"
    DATA_YAML = DATASET_DIR / "data.yaml"

    if not CLASSES_TXT.exists():
      print("classes.txt does not exist.")
      exit(0)

    with open(CLASSES_TXT, "r") as f:
      classes = [line.strip() for line in f.readlines()]

    dataYamlContent = \
f"""
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
"""

    for i, name in enumerate(classes):
      dataYamlContent += f"  {i}: {name}\n"

    with open(DATA_YAML, "w") as f:
      f.write(dataYamlContent)

    print(f"YAML file generated at {DATA_YAML.resolve()}")

  # ------------------------------ Tool functions ------------------------------ #
  def load_items(self):
    items = []

    for item in os.listdir(self.input_dir):
      if not item.endswith('.png'):
        return
      
      # 使用PIL讀取圖片以保持透明度
      item_path = os.path.join(self.input_dir, item)
      print(f"Load item image from: {item_path}")
      pil_img = Image.open(item_path)

      # 確保圖片是RGBA模式
      if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')

      # 轉換為numpy數組
      img = np.array(pil_img)

      # 使用原始檔名（不含副檔名）作為class name
      class_name = os.path.splitext(item)[0]
      items.append((class_name, img))
    
    return items
  
  def rotate_image(self, image, angle):
    # 獲取圖片中心點
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 計算旋轉後的邊界框
    cos = np.abs(np.cos(np.radians(angle)))
    sin = np.abs(np.sin(np.radians(angle)))
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # 調整旋轉矩陣以確保圖片完整
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 執行旋轉，使用INTER_CUBIC插值方法，並保持透明度
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    # 使用閾值處理來去除毛邊
    alpha = rotated[:, :, 3]
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    rotated[:, :, 3] = alpha
    
    # 找到非透明區域的邊界
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 裁剪圖片，只保留有內容的部分
    rotated = rotated[rmin:rmax+1, cmin:cmax+1]
    
    return rotated

  def generate_image(self, max_overlap=0.2, max_attempts=50):
    h, w = self.output_size[1], self.output_size[0]
    reserved_edge = {'top': 0, 'bottom': 0, 'left': 0}

    # 1. 建立白底背景
    background = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 2. 加入邊界干擾區塊
    for side in ['top', 'bottom', 'left']:
      if random.random() > 0.7:
        continue

      thickness = random.randint(5, 25)
      reserved_edge[side] = thickness
      patch = np.full((h, w, 3), np.random.randint(160, 220), dtype=np.uint8)

      for _ in range(random.randint(5, 25)):
        rect_color = np.random.randint(40, 160)
        color = (rect_color,) * 3
        rh, rw = random.randint(10, 50), random.randint(10, 50)

        if side == 'top':
          rh = min(rh, thickness - 1)
          ry = random.randint(0, thickness - rh)
          rx = random.randint(0, w - rw)
        elif side == 'bottom':
          rh = min(rh, thickness - 1)
          ry = random.randint(h - thickness, h - rh)
          rx = random.randint(0, w - rw)
        elif side == 'left':
          rw = min(rw, thickness - 1)
          ry = random.randint(0, h - rh)
          rx = random.randint(0, thickness - rw)

        patch[ry:ry+rh, rx:rx+rw] = color

      if side == 'top':
        background[:thickness] = patch[:thickness]
      elif side == 'bottom':
        background[-thickness:] = patch[-thickness:]
      elif side == 'left':
        background[:, :thickness] = patch[:, :thickness]

    # 3. 放置隨機圖案物件
    placed_objects = []
    final_objects = []
    overlap_counts = np.zeros((h, w), dtype=np.uint8)
    num_items = min(random.randint(4, 7), len(self.items))
    selected_items = random.choices(self.items * 5, k=num_items) # 提高重複 item 機率

    for class_name, item_img in selected_items:
      # 隨機縮放
      scale = random.uniform(0.08, 0.4)
      new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
      resized = cv2.resize(item_img, new_size, interpolation=cv2.INTER_CUBIC)

      # 去毛邊
      alpha = resized[:, :, 3]
      _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
      resized[:, :, 3] = alpha

      # 隨機旋轉
      angle = random.uniform(0, 360)
      rotated = self.rotate_image(resized, angle)
      rotated_height, rotated_width = rotated.shape[:2]

      rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (random.choice([3, 5, 7, 13, 15]),) * 2, 0)

      # 取得當前物件的實際遮罩（忽略完全透明的區域）   
      alpha_mask = rotated[:, :, 3] > 0

      # 隨機位置嘗試
      attempt = 0
      while attempt < max_attempts:
        x_min = reserved_edge['left']
        y_min = reserved_edge['top']
        x_max = w - rotated_width
        y_max = h - reserved_edge['bottom'] - rotated_height
        if x_max <= x_min or y_max <= y_min:
          break  # 沒空間了

        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        current_bbox = [x, y, x + rotated_width, y + rotated_height]

        temp_overlap = overlap_counts.copy()
        temp_overlap[y:y+rotated_height, x:x+rotated_width][alpha_mask] += 1
        if np.any(temp_overlap > 4):
          attempt += 1
          continue

        placement_valid = True
        for placed_obj in placed_objects:
          pb = placed_obj['bbox']
          inter_x1 = max(current_bbox[0], pb[0])
          inter_y1 = max(current_bbox[1], pb[1])
          inter_x2 = min(current_bbox[2], pb[2])
          inter_y2 = min(current_bbox[3], pb[3])

          if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            area2 = (pb[2] - pb[0]) * (pb[3] - pb[1])
            if intersection_area / area1 > max_overlap or intersection_area / area2 > max_overlap:
              placement_valid = False
              break

        if not placement_valid:
          attempt += 1
          continue

        overlap_counts[y:y+rotated_height, x:x+rotated_width][alpha_mask] += 1
        for c in range(3):
          background[y:y+rotated_height, x:x+rotated_width, c][alpha_mask] = rotated[:, :, c][alpha_mask]

        placed_objects.append({'class_id': self.class_to_id[class_name], 'bbox': tuple(current_bbox)})
        final_objects.append({'class_id': self.class_to_id[class_name], 'bbox': tuple(current_bbox)})
        break
      else:
        print(f"跳過 {class_name}（嘗試超過 {max_attempts} 次）")

    # 4. 整張背景處理：模糊、灰階、暗角
    bg_blur = cv2.GaussianBlur(background, (random.choice([9, 15, 17, 21, 25]),) * 2, 0)
    gray = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    X_kernel = cv2.getGaussianKernel(w, int(w * 0.8))
    Y_kernel = cv2.getGaussianKernel(h, int(h * 0.8))
    vignette = (Y_kernel @ X_kernel.T)
    vignette = 0.5 + 0.5 * vignette / vignette.max()

    for i in range(3):
      bg_gray[:, :, i] = bg_gray[:, :, i] * vignette

    background = np.clip(bg_gray * 0.7, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 4, background.shape).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 5. 建立標註
    annotations = []
    for obj in final_objects:
      x1, y1, x2, y2 = obj['bbox']
      xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
      w_, h_ = x2 - x1, y2 - y1
      annotations.append(f"{obj['class_id']} {xc/w:.6f} {yc/h:.6f} {w_/w:.6f} {h_/h:.6f}")

    return background, annotations

  def generate_debug_images(self):
    id_to_class = {v: k for k, v in self.class_to_id.items()}

    for filename in os.listdir(self.image_dir):
      if not filename.endswith('.png'):
        continue

      image_path = os.path.join(self.image_dir, filename)
      label_path = os.path.join(self.label_dir, filename.replace('.png', '.txt'))
      output_path = os.path.join(self.debug_dir, filename)

      image = cv2.imread(image_path)
      height, width = image.shape[:2]

      if not os.path.exists(label_path):
        print(f"找不到標註檔案: {label_path}")
        continue

      with open(label_path, 'r') as f:
        lines = f.readlines()

      for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])
        
        # 轉換為 pixel 座標
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # 畫框與類別文字
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = id_to_class.get(class_id, str(class_id))
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

      # 儲存畫好框的圖片
      cv2.imwrite(output_path, image)

  # Move files
  def move_files(self, files, subset):
    DATASET_DIR = Path(self.dataset_dir)
    LABELS_DIR = DATASET_DIR / "labels"

    for img_file in files:
      # Move image
      shutil.move(str(img_file), str(DATASET_DIR / 'images' / subset / img_file.name))
      
      # Move corresponding label
      label_file = LABELS_DIR / f"{img_file.stem}.txt"
      if label_file.exists():
        shutil.move(str(label_file), str(DATASET_DIR / 'labels' / subset / label_file.name))
