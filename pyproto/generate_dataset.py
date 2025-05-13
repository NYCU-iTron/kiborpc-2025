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
output_size = (640, 640)

def rotate_image(image, angle):
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
    background = np.ones((output_size[1], output_size[0], 4), dtype=np.uint8) * 255
    
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
            
            # 對 item 做高斯模糊
            item_blur_kernel = random.choice([3, 5, 7])
            rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)
            
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
    # 1. 產生白色背景
    background = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

    # 2. 隨機選 1~4 個 item
    available_items = len(items)
    if available_items < 2:
        print(f"警告：只有{available_items}個物品可用，無法生成包含多個物品的圖片")
        return background, []
    num_items = min(random.randint(1, 5), available_items)
    item_pool = items * 5  # 提高重複 item 機率
    selected_items = random.choices(item_pool, k=num_items)

    annotations = []
    bboxes = []  # store existing bbox in pixel coords
    original_boxes = []  # 儲存原始邊界框

    for class_name, item_img in selected_items:
        # 隨機縮放
        scale = random.uniform(0.1, 0.4)
        new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
        resized = cv2.resize(item_img, new_size, interpolation=cv2.INTER_CUBIC)
        # 使用閾值處理來去除毛邊
        alpha = resized[:, :, 3]
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        resized[:, :, 3] = alpha
        # 隨機旋轉
        angle = random.uniform(0, 360)
        rotated = rotate_image(resized, angle)
        rotated_height, rotated_width = rotated.shape[:2]
        # 對 item 做高斯模糊
        item_blur_kernel = random.choice([3, 5, 7])
        rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)
        # 隨機位置
        attempt = 0
        while attempt < max_attempts:
            x = random.randint(0, output_size[0] - rotated_width)
            y = random.randint(0, output_size[1] - rotated_height)
            new_bbox = (x, y, x + rotated_width, y + rotated_height)
            if (x >= 0 and y >= 0 and x + rotated_width <= output_size[0] and y + rotated_height <= output_size[1]):
                overlap_ok = True
                for existing in bboxes:
                    iou = bbox_iou(existing, new_bbox)
                    if iou > max_overlap:
                        overlap_ok = False
                        break
                if overlap_ok:
                    break
            attempt += 1
        if attempt >= max_attempts:
            print(f"跳過 {class_name}（嘗試超過 {max_attempts} 次）")
            continue
        # 將物品貼到背景上（只覆蓋 alpha > 0 的區域）
        alpha_mask = rotated[:, :, 3] > 0
        for c in range(3):
            background[y:y+rotated_height, x:x+rotated_width, c][alpha_mask] = rotated[:, :, c][alpha_mask]
        # 儲存原始邊界框
        original_boxes.append({
            'class_id': class_to_id[class_name],
            'box': (x, y, x + rotated_width, y + rotated_height)
        })
    # 3. 整張圖做高斯模糊、灰階、暗角
    blur_kernel = random.choice([7, 9, 11])
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
    # 4. 產生標註
    annotations = []
    for box_data in original_boxes:
        x1, y1, x2, y2 = box_data['box']
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        x_center = x_center / output_size[0]
        y_center = y_center / output_size[1]
        width = width / output_size[0]
        height = height / output_size[1]
        annotations.append(f"{box_data['class_id']} {x_center} {y_center} {width} {height}")
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
    item_pool = items * 5  # 提高重複 item 機率
    selected_items = random.choices(item_pool, k=num_items)

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
if False:
    last_image_count = generate_rotated_dataset()
else: 
    last_image_count = 0

# 生成隨機組合的圖片
# for i in range(2200):
#     image, annotations = generate_image_two_stage()
    
#     # 保存圖片
#     cv2.imwrite(f'../assets/dataset/images/image_{last_image_count + i:04d}.png', image)
    
#     # 保存標註
#     with open(f'../assets/dataset/labels/image_{last_image_count + i:04d}.txt', 'w') as f:
#         f.write('\n'.join(annotations))

for i in range(2000):

    image, annotations = generate_image()
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