import os
import cv2
import numpy as np
from PIL import Image
import random
import math


# 創建輸出目錄
os.makedirs('../assets/dataset/images', exist_ok=True)
os.makedirs('../assets/dataset/labels', exist_ok=True)

# 讀取所有物品圖片
items = []
class_names = []
class_to_id = {}  # 用於儲存class_name到class_id的映射
input_dir = '../assets/item_template_images/processed'
print(f"正在從 {input_dir} 讀取圖片...")
print(f"目錄內容: {os.listdir(input_dir)}")

# 首先收集所有class_names並建立映射
class_to_id = {
    'coin': 21, 
    'compass': 22, 
    'coral': 23, 
    'crystal': 11, 
    'diamond': 12, 
    'emerald': 13, 
    'fossil': 24, 
    'key': 25, 
    'letter': 26, 
    'shell': 27, 
    'treasure_box': 28
}


for item in os.listdir(input_dir):
    if item.endswith('.png'):
        class_name = os.path.splitext(item)[0]
        if class_name not in class_to_id:
            class_to_id[class_name] = len(class_to_id)  # 自動分配ID
            class_names.append(class_name)

print(f"找到的類別: {class_to_id}")

# 然後讀取圖片
for item in os.listdir(input_dir):
    if item.endswith('.png'):
        # 使用PIL讀取圖片以保持透明度
        item_path = os.path.join(input_dir, item)
        print(f"處理圖片: {item_path}")
        pil_img = Image.open(item_path)
        # 確保圖片是RGBA模式
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        # 轉換為numpy數組
        img = np.array(pil_img)
        # 使用原始檔名（不含副檔名）作為class name
        class_name = os.path.splitext(item)[0]
        items.append((class_name, img))

# 設置輸出圖片大小
output_size = (1280, 960)

def rotate_image(image, angle):
    # 獲取圖片中心點
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 創建旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 執行旋轉，使用INTER_CUBIC插值方法，並保持透明度
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    # 使用閾值處理來去除毛邊
    alpha = rotated[:, :, 3]
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    rotated[:, :, 3] = alpha
    
    return rotated

def bbox_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box2[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def generate_single_item_image(item_img, class_name, scale=0.4):
    # 創建白色背景（RGBA）
    env_path = random.choice(os.listdir('../assets/env'))
    env_img = Image.open(os.path.join('../assets/env', env_path)).convert('RGBA')
    env_img = env_img.resize(output_size, resample=Image.LANCZOS)
    background = np.array(env_img)
    
    # 縮放圖片，使用INTER_CUBIC插值方法
    new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
    resized = cv2.resize(item_img, new_size, interpolation=cv2.INTER_CUBIC)
    
    # 使用閾值處理來去除毛邊
    alpha = resized[:, :, 3]
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    resized[:, :, 3] = alpha
    
    # 計算中心位置
    x = (output_size[0] - new_size[0]) // 2
    y = (output_size[1] - new_size[1]) // 2
    
    # 將物品放置到背景上
    alpha_mask = resized[:, :, 3] > 0
    for c in range(3):  # 只處理RGB通道
        background[y:y+new_size[1], x:x+new_size[0], c][alpha_mask] = resized[:, :, c][alpha_mask]
    # 處理alpha通道
    background[y:y+new_size[1], x:x+new_size[0], 3][alpha_mask] = resized[:, :, 3][alpha_mask]
    
    # 生成YOLO格式的標註
    class_id = class_to_id[class_name]
    x_center = (x + new_size[0]/2) / output_size[0]
    y_center = (y + new_size[1]/2) / output_size[1]
    width = new_size[0] / output_size[0]
    height = new_size[1] / output_size[1]
    
    annotation = f"{class_id} {x_center} {y_center} {width} {height}"
    
    return background, annotation

def generate_rotated_dataset():
    print("開始生成8個方位的旋轉圖片...")
    start_count = len([f for f in os.listdir('../assets/dataset/images') if f.startswith('image_')])
    image_count = start_count
    
    for class_name, item_img in items:
        # 生成8個方位的旋轉圖片
        for angle in range(0, 360, 8):  # 0, 45, 90, 135, 180, 225, 270, 315
            # 旋轉圖片
            rotated = rotate_image(item_img, angle)
            
            # 生成單一物品的圖片
            image, annotation = generate_single_item_image(rotated, class_name)
            
            # 保存圖片，使用與現有數據集相同的命名格式
            cv2.imwrite(f'../assets/dataset/images/image_{image_count:04d}.png', image)
            
            # 保存標註，使用與現有數據集相同的命名格式
            with open(f'../assets/dataset/labels/image_{image_count:04d}.txt', 'w') as f:
                f.write(annotation)
            
            image_count += 1
            print(f"已生成 {class_name} 的 {angle} 度旋轉圖片")
    
    print(f"共生成 {image_count - start_count} 張旋轉圖片")
    return image_count  # 返回最後的圖片計數，供後續使用

def generate_image(max_overlap=0.7, max_attempts=50):
    # 創建白色背景（RGBA）
    env_path = random.choice(os.listdir('../assets/env'))
    env_img = Image.open(os.path.join('../assets/env', env_path)).convert('RGBA')
    env_img = env_img.resize(output_size, resample=Image.LANCZOS)
    background = np.array(env_img)
    
    # 隨機選擇2-4個物品，但確保不超過可用的物品數量
    available_items = len(items)
    if available_items < 2:
        print(f"警告：只有{available_items}個物品可用，無法生成包含多個物品的圖片")
        return background, []
    
    num_items = min(random.randint(1, 4), available_items)
    selected_items = random.sample(items, num_items)
    
    annotations = []
    bboxes = []  # store existing bbox in pixel coords

    for class_name, item_img in selected_items:
        # 隨機縮放
        scale = random.uniform(0.1, 0.5)
        new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
        resized = cv2.resize(item_img, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 使用閾值處理來去除毛邊
        alpha = resized[:, :, 3]
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        resized[:, :, 3] = alpha
        
        # 隨機旋轉
        angle = random.uniform(0, 360)
        rotated = rotate_image(resized, angle)
        
        # 隨機位置
        attempt = 0
        while attempt < max_attempts:
            x = random.randint(0, output_size[0] - new_size[0])
            y = random.randint(0, output_size[1] - new_size[1])
            new_bbox = (x, y, x + new_size[0], y + new_size[1])

            overlap_ok = True
            for existing in bboxes:
                iou = bbox_iou(existing, new_bbox)
                if iou > max_overlap:
                    overlap_ok = False
                    break

            if overlap_ok:
                break  # accept this placement
            attempt += 1

        if attempt >= max_attempts:
            print(f"跳過 {class_name}（嘗試超過 {max_attempts} 次）")
            continue

        # 將物品放置到背景上
        alpha_mask = rotated[:, :, 3] > 0
        for c in range(3):  # 只處理RGB通道
            background[y:y+new_size[1], x:x+new_size[0], c][alpha_mask] = rotated[:, :, c][alpha_mask]
        # 處理alpha通道
        background[y:y+new_size[1], x:x+new_size[0], 3][alpha_mask] = rotated[:, :, 3][alpha_mask]
        
        # 生成YOLO格式的標註
        # YOLO格式：<class_id> <x_center> <y_center> <width> <height>
        class_id = class_to_id[class_name]  # 使用數字ID
        x_center = (x + new_size[0]/2) / output_size[0]
        y_center = (y + new_size[1]/2) / output_size[1]
        width = new_size[0] / output_size[0]
        height = new_size[1] / output_size[1]
        
        annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return background, annotations

def paste_rgba_image(background, overlay, x_offset, y_offset):
    h, w = overlay.shape[:2]
    alpha_mask = overlay[:, :, 3] > 0

    for c in range(3):  # R, G, B
        background[y_offset:y_offset+h, x_offset:x_offset+w, c][alpha_mask] = \
            overlay[:, :, c][alpha_mask]
    background[y_offset:y_offset+h, x_offset:x_offset+w, 3][alpha_mask] = \
        overlay[:, :, 3][alpha_mask]

    return background

def generate_image_two_stage(max_overlap=0.5, max_attempts=50):
    output_size = (1280, 960)

    # === Stage 1: 貼 item 到 stage1 背景 ===
    stage1_bg_path = "../assets/background.png"
    stage1_img = Image.open(stage1_bg_path).convert('RGBA')

    # 隨機縮放 stage1 背景
    scale_factor = random.uniform(0.1, 0.3)
    stage1_size = (int(output_size[0] * scale_factor), int(output_size[1] * scale_factor))
    stage1_img = stage1_img.resize(stage1_size, resample=Image.LANCZOS)
    stage1_np = np.array(stage1_img)

    # 選 1～4 個 item，貼到 stage1 背景的「左 70%、中間 70%」區域
    available_items = len(items)
    if available_items < 1:
        print("沒有足夠的 items")
        return stage1_np, []

    num_items = min(random.randint(1, 4), available_items)
    selected_items = random.sample(items, num_items)

    annotations = []
    bboxes = []

    for class_name, item_img in selected_items:
        scale = random.uniform(0.4, 0.7) * scale_factor
        new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
        resized = cv2.resize(item_img, new_size, interpolation=cv2.INTER_CUBIC)

        alpha = resized[:, :, 3]
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        resized[:, :, 3] = alpha

        rotated = rotate_image(resized, random.uniform(0, 360))

        attempt = 0
        while attempt < max_attempts:
            x = random.randint(0, int(stage1_size[0] * 0.7) - new_size[0])
            y = random.randint(int(stage1_size[1] * 0.15), int(stage1_size[1] * 0.85) - new_size[1])
            new_bbox = (x, y, x + new_size[0], y + new_size[1])

            # 檢查是否與現有物品重疊過多
            if all(bbox_iou(existing, new_bbox) <= max_overlap for existing in bboxes):
                bboxes.append(new_bbox)
                break
            attempt += 1

        if attempt >= max_attempts:
            print(f"跳過 {class_name}（重疊過多或嘗試次數超過 {max_attempts} 次）")
            continue

        stage1_np = paste_rgba_image(stage1_np, rotated, x, y)

        # bbox for YOLO annotation (for final env image, so need to compute after placement in stage2)
        annotations.append((class_to_id[class_name], x, y, new_size[0], new_size[1], stage1_size))

    # === Stage 2: 把 stage1 圖貼到最終背景 ===
    env_path = os.path.join('../assets/env', random.choice(os.listdir('../assets/env')))
    env_img = Image.open(env_path).convert('RGBA')
    env_img = env_img.resize(output_size, resample=Image.LANCZOS)
    env_np = np.array(env_img)

    # 貼到環境圖的中間
    x_offset = (output_size[0] - stage1_size[0]) // 2
    y_offset = (output_size[1] - stage1_size[1]) // 2

    # # 隨機計算第一層貼到第二層的位置
    # x_offset = random.randint(0, output_size[0] - stage1_size[0])
    # y_offset = random.randint(0, output_size[1] - stage1_size[1])

    env_np = paste_rgba_image(env_np, stage1_np, x_offset, y_offset)

    # === 調整 bounding box 到環境圖尺寸 ===
    final_annotations = []
    for class_id, x, y, w, h, stage1_dim in annotations:
        x += x_offset
        y += y_offset
        x_center = (x + w / 2) / output_size[0]
        y_center = (y + h / 2) / output_size[1]
        width = w / output_size[0]
        height = h / output_size[1]
        final_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return env_np, final_annotations





# 生成旋轉數據集
if True:
    last_image_count = generate_rotated_dataset()
else: 
    last_image_count = 0

# 生成隨機組合的圖片
for i in range(2200):
    image, annotations = generate_image_two_stage()
    
    # 保存圖片
    cv2.imwrite(f'../assets/dataset/images/image_{last_image_count + i:04d}.png', image)
    
    # 保存標註
    with open(f'../assets/dataset/labels/image_{last_image_count + i:04d}.txt', 'w') as f:
        f.write('\n'.join(annotations))

# 生成classes.txt，使用數字ID和對應的class_name
with open('../assets/dataset/classes.txt', 'w') as f:
    for class_name in class_names:
        class_id = class_to_id[class_name]
        f.write(f'{class_id} {class_name}\n')

print(f"找到 {len(items)} 個物品: {class_names}")
if len(items) < 2:
    print("錯誤：需要至少2個物品才能生成數據集")
    exit(1)

print("數據集生成完成！") 