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

    self.items = self.load_items()

  def generate_data(self, total_image_count):
    count = 0

    for i in range(1, total_image_count+1):
      r = random.random()

      if r < 0.1:
        max_overlap = 0.7
      elif r > 0.1 and r < 0.5:
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

    # split 80% train, 20% val
    split_idx = int(0.9 * len(image_files))
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
    # 1. 產生白色背景
    background = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255

    # 2. 隨機選 1~4 個 item
    available_items = len(self.items)

    num_items = min(random.randint(3, 7), available_items)
    item_pool = self.items * 5  # 提高重複 item 機率
    selected_items = random.choices(item_pool, k=num_items)

    placed_objects = []  # 以順序存放成功放置的物件資訊
    overlap_counts = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.uint8)  # 追蹤重疊次數
    final_objects = []  # 儲存最終的物件資訊（用於標籤）

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

      # 模糊 item
      item_blur_kernel = random.choice([3, 5, 7, 13, 15])
      rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)

      # 取得當前物件的實際遮罩（忽略完全透明的區域）
      alpha_mask = rotated[:, :, 3] > 0
      
      # 隨機位置嘗試
      attempt = 0
      while attempt < max_attempts:
        x = random.randint(0, self.output_size[0] - rotated_width)
        y = random.randint(0, self.output_size[1] - rotated_height)
        
        if (x < 0 or y < 0 or x + rotated_width > self.output_size[0] or y + rotated_height > self.output_size[1]):
          attempt += 1
          continue

        # 使用邊界框計算
        current_bbox = [x, y, x + rotated_width, y + rotated_height]
        current_img = rotated
        current_mask = alpha_mask
        
        # 檢查重疊次數限制
        temp_overlap = overlap_counts.copy()
        temp_overlap[y:y+rotated_height, x:x+rotated_width][alpha_mask] += 1
        
        if np.any(temp_overlap > 4):      
          attempt += 1
          continue
        
        # 檢查與已放置物件的邊界框重疊情況
        placement_valid = True
        for placed_obj in placed_objects:
          placed_bbox = placed_obj['bbox']
          
          # 計算邊界框交集
          inter_x1 = max(current_bbox[0], placed_bbox[0])
          inter_y1 = max(current_bbox[1], placed_bbox[1])
          inter_x2 = min(current_bbox[2], placed_bbox[2])
          inter_y2 = min(current_bbox[3], placed_bbox[3])
          
          # 如果沒有交集，繼續檢查下一個物件
          if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue
          
          # 計算交集面積
          intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
          
          # 計算兩個邊界框的面積
          bbox1_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
          bbox2_area = (placed_bbox[2] - placed_bbox[0]) * (placed_bbox[3] - placed_bbox[1])
          
          # 計算重疊率
          overlap_ratio1 = intersection_area / bbox1_area
          overlap_ratio2 = intersection_area / bbox2_area
          
          # 如果任一重疊率超過閾值，且 max_overlap > 0.4，執行裁切
          if (overlap_ratio1 > max_overlap or overlap_ratio2 > max_overlap) and max_overlap > 0.4:
            # 裁切邏輯：縮減當前邊界框以減少重疊
            crop_success = False
            min_size = min(rotated_width, rotated_height) * 0.5  # 最小尺寸限制
            
            # 計算裁切後的邊界框（優先裁切右邊或下邊）
            crop_x1 = inter_x1 if inter_x1 > current_bbox[0] else current_bbox[0]
            crop_y1 = inter_y1 if inter_y1 > current_bbox[1] else current_bbox[1]
            crop_x2 = inter_x2 if inter_x2 < current_bbox[2] else current_bbox[2]
            crop_y2 = inter_y2 if inter_y2 < current_bbox[3] else current_bbox[3]
            
            # 嘗試裁切右邊或下邊
            if (crop_x2 - current_bbox[0]) >= min_size:
              crop_x2 = max(current_bbox[0], inter_x1 - 1)  # 裁切到交集左邊
            elif (crop_y2 - current_bbox[1]) >= min_size:
              crop_y2 = max(current_bbox[1], inter_y1 - 1)  # 裁切到交集上邊
            else:
              placement_valid = False
              break
            
            # 更新裁切後的圖像和遮罩
            crop_width = crop_x2 - current_bbox[0]
            crop_height = crop_y2 - current_bbox[1]

            if crop_width > 0 and crop_height > 0:
              crop_x_offset = current_bbox[0] - x
              crop_y_offset = current_bbox[1] - y
              current_img = current_img[crop_y_offset:crop_y_offset+crop_height, crop_x_offset:crop_x_offset+crop_width]
              current_mask = current_mask[crop_y_offset:crop_y_offset+crop_height, crop_x_offset:crop_x_offset+crop_width]
              current_bbox = [x, y, x + crop_width, y + crop_height]
              crop_success = True
            
            if not crop_success:
              placement_valid = False
              break
            
            # 重新計算重疊比例
            bbox1_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            overlap_ratio1 = intersection_area / bbox1_area if bbox1_area > 0 else 1.0
            if overlap_ratio1 > max_overlap:
              placement_valid = False
              break
    
          elif overlap_ratio1 > max_overlap or overlap_ratio2 > max_overlap:
            placement_valid = False
            break
      
        if placement_valid:
          # 更新重疊計數圖
          overlap_counts[y:y+current_img.shape[0], x:x+current_img.shape[1]][current_mask] += 1
          
          # 貼 item 到背景
          for c in range(3):
            background[y:y+current_img.shape[0], x:x+current_img.shape[1], c][current_mask] = current_img[:, :, c][current_mask]
          
          # 儲存物件資訊到 placed_objects
          placed_objects.append({
            'class_id': self.class_to_id[class_name],
            'bbox': tuple(current_bbox),
            'mask': current_mask,
            'x': x,
            'y': y,
            'width': current_img.shape[1],
            'height': current_img.shape[0]
          })
          
          # 將邊界框加入 final_objects 用於標籤
          final_objects.append({
            'class_id': self.class_to_id[class_name],
            'bbox': tuple(current_bbox)
          })
          
          break
        
        attempt += 1

        if attempt >= max_attempts:
          print(f"跳過 {class_name}（嘗試超過 {max_attempts} 次）")
          break

    # 3. 整張圖做高斯模糊、灰階、暗角
    blur_kernel = random.choice([9, 15, 17, 21, 25])
    bg_blur = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), 0)
    gray = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # vignette
    rows, cols = bg_gray.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, int(cols*0.8))
    Y_resultant_kernel = cv2.getGaussianKernel(rows, int(rows*0.8))
    vignette_mask = Y_resultant_kernel * X_resultant_kernel.T
    vignette_mask = vignette_mask / vignette_mask.max()
    vignette_mask = 0.5 + 0.5 * vignette_mask  # 讓中心和角落差異更小

    for i in range(3):
      bg_gray[:, :, i] = bg_gray[:, :, i] * vignette_mask

    # 整體降低亮度
    background = np.clip(bg_gray * 0.7, 0, 255).astype(np.uint8)

    # 加入隨機燥點
    noise = np.random.normal(0, 4, background.shape).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 產生標註
    annotations = []
    for obj in final_objects:
      x1, y1, x2, y2 = obj['bbox']
      x_center = (x1 + x2) / 2
      y_center = (y1 + y2) / 2
      width = x2 - x1
      height = y2 - y1
      x_center = x_center / self.output_size[0]
      y_center = y_center / self.output_size[1]
      width = width / self.output_size[0]
      height = height / self.output_size[1]
      annotations.append(f"{obj['class_id']} {x_center} {y_center} {width} {height}")

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
