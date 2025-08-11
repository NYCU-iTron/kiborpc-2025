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
input_dir = '../assets/item_images'
print(f"正在從 {input_dir} 讀取圖片...")
print(f"目錄內容: {os.listdir(input_dir)}")


class_to_id = {
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


for item in os.listdir(input_dir):
    if item.endswith('.png'):
        class_name = os.path.splitext(item)[0]
        if class_name not in class_to_id:
            class_to_id[class_name] = len(class_to_id)  # 自動分配ID
            class_names.append(class_name)

print(f"找到的類別: {class_to_id}")

# 然後讀取圖片
for item in os.listdir(input_dir):
    if item.endswith('.webp'):
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
output_size = (160, 160)

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

def apply_full_image_augmentations(image, original_boxes_data_list):
    """
    對整張圖片應用 Shear 和 亮度/對比度 增強, 並更新邊界框。
    :param image: numpy array, 輸入圖像 (RGB 或 RGBA)
    :param original_boxes_data_list: list of dicts, e.g., [{'class_id': id, 'box': (x1,y1,x2,y2)}, ...]
    :return: tuple (augmented_image, updated_boxes_data_list)
    """
    # 1. Shear
    shear_factor = random.uniform(-0.05, 0.05)
    # M_shear for horizontal shear: x' = x + shear_factor * y, y' = y
    M_shear = np.array([[1, shear_factor, 0],
                        [0, 1,            0]], dtype=np.float32)
    
    h_img, w_img = image.shape[:2]
    is_rgba = image.shape[2] == 4

    img_rgb_part = image[:,:,:3]
    alpha_part_sheared = None

    if is_rgba:
        # Shear alpha channel
        alpha_part_sheared = cv2.warpAffine(image[:,:,3], M_shear, (w_img, h_img), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Shear RGB part
    img_rgb_sheared = cv2.warpAffine(img_rgb_part, M_shear, (w_img, h_img))

    # 2. Adjust Bounding Boxes for Shear
    updated_boxes_data_list = []
    if original_boxes_data_list:
        for box_data in original_boxes_data_list:
            class_id = box_data['class_id']
            x1, y1, x2, y2 = box_data['box'] # x2, y2 are end-exclusive coords in some contexts, but here they are inclusive max pixel coord values for width/height calc
            
            # Define the 4 corners of the original bounding box
            # (top-left, top-right, bottom-left, bottom-right)
            corners = np.array([
                [x1, y1], [x2, y1],
                [x1, y2], [x2, y2]
            ])

            # Apply shear transformation to each corner: x_new = x_old + shear_factor * y_old; y_new = y_old
            transformed_corners = np.array([
                [c[0] + shear_factor * c[1], c[1]] for c in corners
            ])
            
            # Calculate the new axis-aligned bounding box
            new_x1 = np.min(transformed_corners[:, 0])
            new_y1 = np.min(transformed_corners[:, 1]) # Should be original y1
            new_x2 = np.max(transformed_corners[:, 0])
            new_y2 = np.max(transformed_corners[:, 1]) # Should be original y2

            # Clip to image boundaries
            new_x1 = np.clip(new_x1, 0, w_img)
            new_y1 = np.clip(new_y1, 0, h_img)
            new_x2 = np.clip(new_x2, 0, w_img)
            new_y2 = np.clip(new_y2, 0, h_img)

            # Ensure x1 <= x2 and y1 <= y2
            if new_x1 > new_x2: new_x1, new_x2 = new_x2, new_x1 
            if new_y1 > new_y2: new_y1, new_y2 = new_y2, new_y1
            
            updated_boxes_data_list.append({'class_id': class_id, 'box': (new_x1, new_y1, new_x2, new_y2)})

    # 3. Brightness/Contrast on the sheared RGB part
    img_rgb_adjusted = img_rgb_sheared.copy() 
    brightness_factor = random.uniform(0.9, 1.3)
    contrast_factor = random.uniform(0.9, 1.1)
    img_rgb_adjusted = cv2.convertScaleAbs(img_rgb_adjusted, alpha=contrast_factor, beta=(brightness_factor - 1) * 128)

    # 4. Reconstruct final image
    if is_rgba and alpha_part_sheared is not None:
        final_image = cv2.merge((img_rgb_adjusted, alpha_part_sheared))
    else:
        final_image = img_rgb_adjusted
        
    return final_image, updated_boxes_data_list

def bbox_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) # Corrected box2[1] to box1[1]
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def generate_image(max_overlap=None, max_attempts=50, force_this_image_to_overlap=False, num_items_override=None):
    if max_overlap is None:
        max_overlap_val = random.uniform(0.5, 0.8)
    else:
        max_overlap_val = max_overlap
    
    if force_this_image_to_overlap:
        # 如果被要求強制重疊，我們將使用一個更小的 max_overlap 值，並且可能增加物品數量
        max_overlap_val = random.uniform(0.05, 0.25) 
        if num_items_override is None and random.random() < 0.7: # 70% 的機率增加物品數量
            num_items_override = random.randint(2,5) # 假設 items 數量足夠

    # 1. 產生白色背景
    background = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

    # 2. 隨機選 1~4 個 item
    available_items = len(items)
    if available_items < 2:
        print(f"警告：只有{available_items}個物品可用，無法生成包含多個物品的圖片")
        return background, [], False
    
    if num_items_override is not None:
        num_items_to_select = min(num_items_override, available_items)
    else:
        num_items_to_select = min(random.choice([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5]), available_items)
    
    if force_this_image_to_overlap and num_items_to_select < 2 and available_items >=2:
        num_items_to_select = 2 # 強制重疊時至少要有兩個物品
        
    item_pool = items * 5  # 提高重複 item 機率
    selected_items = random.choices(item_pool, k=num_items_to_select)

    annotations = []
    bboxes = []  # store existing bbox in pixel coords
    original_boxes = []  # 儲存原始邊界框
    has_overlapped_item = False

    for class_name, item_img in selected_items:
        # 隨機縮放
        scale = random.uniform(0.05, 0.15)
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

        # Item-specific blur is still applied before placing on background
        item_blur_kernel = random.choice(list(range(1, 7, 2))) 
        rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)
        
        # 隨機位置
        attempt = 0
        while attempt < max_attempts:
            x = random.randint(0, output_size[0] - rotated_width)
            y = random.randint(0, output_size[1] - rotated_height)
            new_bbox = (x, y, x + rotated_width, y + rotated_height)
            if (x >= 0 and y >= 0 and x + rotated_width <= output_size[0] and y + rotated_height <= output_size[1]):
                overlap_ok = True
                if len(bboxes) > 0: # 只有當畫布上已經有物件時才檢查重疊
                    for existing_idx, existing in enumerate(bboxes):
                        iou = bbox_iou(existing, new_bbox)
                        if iou > max_overlap_val: # If IoU is TOO HIGH, placement is not OK
                            overlap_ok = False
                            break
                        elif iou > 0.01: # <<<< 修改點：只有當 IoU > 0.01 才視為有效重疊並計數
                            has_overlapped_item = True 
                if overlap_ok:
                    break # Placement is OK
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
    # After all items are placed, apply augmentations to the entire image
    if len(original_boxes) > 0:
        # Ensure background is RGB before passing to augmentations if it might be RGBA
        # In generate_image, background is initialized as 3-channel np.ones * 255
        background, original_boxes = apply_full_image_augmentations(background, original_boxes)
    else:
        # If no items were placed, still apply brightness/contrast to the plain background
        # but skip shear as it's not meaningful and bbox update is not needed.
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3)
        background = cv2.convertScaleAbs(background, alpha=contrast_factor, beta=(brightness_factor - 1) * 128)

    # 3. 整張圖做高斯模糊、灰階、暗角 (these are applied after shear/brightness)
    blur_kernel = random.choice(list(range(5, 13, 2))) 
    random_blur_kernel = random.choice(list(range(0, 2, 1))) # 從 1, 3, ..., 9 中選擇
    bg_blur = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), random_blur_kernel)
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
    # 整體隨機調整亮度
    brightness_adjust = random.uniform(0.7, 0.9)
    background = np.clip(bg_gray * brightness_adjust, 0, 255).astype(np.uint8)
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
    return background, annotations, has_overlapped_item

def generate_same_item_multiple_instances(item_class_name: str, num_instances: int, max_overlap_val: float = 0.5, max_attempts=50):
    """
    生成包含2-4個相同物品實例的圖像，並可控制它們之間的最大重疊。

    :param item_class_name: 要使用的物品的類別名稱。
    :param num_instances: 要放置的物品實例的數量 (建議 2-4)。
    :param max_overlap_val: 允許的物品之間的最高IoU。
    :param max_attempts: 嘗試放置每個物品的最大次數。
    :return: tuple (augmented_image, annotations_list)
    """
    global items, class_to_id, output_size

    item_img_data = None
    for name, img_data in items:
        if name == item_class_name:
            item_img_data = img_data
            break
    
    if item_img_data is None:
        print(f"錯誤：找不到類別名稱為 '{item_class_name}' 的物品。")
        return None, []

    if not (2 <= num_instances <= 4):
        print(f"警告：建議的物品實例數量為 2 到 4 之間，但收到了 {num_instances}。將繼續處理。")

    # 1. 產生白色背景 (RGB)
    background = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255
    
    annotations = []
    bboxes = []  # 儲存已放置物品的像素座標邊界框 (x1, y1, x2, y2)
    original_boxes_for_augmentation = [] # 儲存用於後續整體增強的原始邊界框數據

    for _ in range(num_instances):
        # 複製物品圖像以進行獨立增強
        current_item_img = item_img_data.copy()

        # 隨機縮放 (與 generate_image 中的範圍一致)
        scale = random.uniform(0.05, 0.15)
        new_size = (int(current_item_img.shape[1] * scale), int(current_item_img.shape[0] * scale))
        if new_size[0] == 0 or new_size[1] == 0: # 避免縮放後尺寸為0
            print(f"警告：物品 '{item_class_name}' 縮放後尺寸過小，跳過此實例。")
            continue
        resized = cv2.resize(current_item_img, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 使用閾值處理來去除毛邊 (針對alpha通道)
        alpha_channel = resized[:, :, 3]
        _, alpha_thresholded = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
        resized[:, :, 3] = alpha_thresholded
        
        # 隨機旋轉
        angle = random.uniform(0, 360)
        rotated = rotate_image(resized, angle) # rotate_image 返回裁剪後的RGBA圖像
        rotated_height, rotated_width = rotated.shape[:2]

        if rotated_height == 0 or rotated_width == 0:
            print(f"警告：物品 '{item_class_name}' 旋轉和裁剪後尺寸為0，跳過此實例。")
            continue

        # Item-specific blur (與 generate_image 中更新後的範圍一致)
        item_blur_kernel = random.choice(list(range(1, 7, 2))) 
        if item_blur_kernel > 0 : # 確保kernel size > 0
             rotated[:, :, :3] = cv2.GaussianBlur(rotated[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)
        
        # 隨機位置並檢查重疊
        attempt = 0
        placed_successfully = False
        while attempt < max_attempts:
            x = random.randint(0, max(0, output_size[0] - rotated_width))
            y = random.randint(0, max(0, output_size[1] - rotated_height))
            new_bbox_coords = (x, y, x + rotated_width, y + rotated_height)
            
            # 確保新物品在邊界內
            if not (x >= 0 and y >= 0 and x + rotated_width <= output_size[0] and y + rotated_height <= output_size[1]):
                attempt +=1
                continue

            overlap_ok = True
            if bboxes: # 只有當畫布上已經有物件時才檢查重疊
                for existing_bbox in bboxes:
                    iou = bbox_iou(existing_bbox, new_bbox_coords)
                    if iou > max_overlap_val:
                        overlap_ok = False
                        break
            
            if overlap_ok:
                bboxes.append(new_bbox_coords)
                placed_successfully = True
                break
            attempt += 1
            
        if not placed_successfully:
            print(f"跳過物品 '{item_class_name}' 的一個實例（嘗試超過 {max_attempts} 次未能找到合適位置）。")
            continue

        # 將物品貼到背景上（只覆蓋 alpha > 0 的區域）
        alpha_mask = rotated[:, :, 3] > 0
        for c in range(3): # RGB通道
            background[y:y+rotated_height, x:x+rotated_width, c][alpha_mask] = rotated[:, :, c][alpha_mask]
        
        # 儲存原始邊界框以進行後續的整體圖像增強
        original_boxes_for_augmentation.append({
            'class_id': class_to_id[item_class_name],
            'box': (x, y, x + rotated_width, y + rotated_height) # x1, y1, x2, y2
        })

    # 如果沒有成功放置任何物品
    if not original_boxes_for_augmentation:
        print(f"警告：未能為物品 '{item_class_name}' 放置任何實例。")
         # 仍然對背景應用一些亮度/對比度變化
        brightness_factor = random.uniform(0.9, 1.3) # 與 apply_full_image_augmentations 中的範圍一致
        contrast_factor = random.uniform(0.9, 1.1)
        background = cv2.convertScaleAbs(background, alpha=contrast_factor, beta=(brightness_factor - 1) * 128)
    else:
        # 對包含所有物品的整張圖片應用Shear和亮度/對比度調整
        # 注意：apply_full_image_augmentations 期望背景是 RGB 或 RGBA
        background, updated_original_boxes = apply_full_image_augmentations(background, original_boxes_for_augmentation)
        original_boxes_for_augmentation = updated_original_boxes


    # 整張圖做高斯模糊、灰階、暗角 (與 generate_image 中更新後的範圍一致)
    blur_kernel_val = random.choice(list(range(5, 13, 2))) 
    if blur_kernel_val > 0:
        # random_blur_sigma = random.choice(list(range(0, 2, 1))) # generate_image 中的 random_blur_kernel
        # 根據 generate_image，sigmaX 可以設為0讓 OpenCV 自動計算，或者是一個小值
        bg_blur = cv2.GaussianBlur(background, (blur_kernel_val, blur_kernel_val), 0) 
    else:
        bg_blur = background.copy()

    gray = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Vignette
    rows, cols = bg_gray.shape[:2]
    if rows > 0 and cols > 0:
        X_resultant_kernel = cv2.getGaussianKernel(cols, int(cols*0.8))
        Y_resultant_kernel = cv2.getGaussianKernel(rows, int(rows*0.8))
        vignette_mask = Y_resultant_kernel * X_resultant_kernel.T
        vignette_mask = vignette_mask / vignette_mask.max()
        vignette_mask = 0.5 + 0.5 * vignette_mask # 讓中心和角落差異更小
        for i in range(3):
            bg_gray[:, :, i] = bg_gray[:, :, i] * vignette_mask
    
    # 整體隨機調整亮度 (與 generate_image 中更新後的範圍一致)
    brightness_adjust = random.uniform(0.7, 0.9)
    final_image = np.clip(bg_gray * brightness_adjust, 0, 255).astype(np.uint8)

    # 產生YOLO格式的標註 (基於增強後更新的邊界框)
    yolo_annotations = []
    for box_data in original_boxes_for_augmentation:
        x1, y1, x2, y2 = box_data['box']
        
        # 確保寬高不為負或零 (可能由於裁切或極端增強導致)
        img_width_px = x2 - x1
        img_height_px = y2 - y1
        if img_width_px <=0 or img_height_px <=0:
            continue # 跳過無效邊界框

        x_center_px = x1 + img_width_px / 2
        y_center_px = y1 + img_height_px / 2
        
        # 歸一化
        x_center_norm = x_center_px / output_size[0]
        y_center_norm = y_center_px / output_size[1]
        width_norm = img_width_px / output_size[0]
        height_norm = img_height_px / output_size[1]
        
        # 確保標註值在 [0, 1] 範圍內
        x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
        y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 0.0, 1.0)
        height_norm = np.clip(height_norm, 0.0, 1.0)

        yolo_annotations.append(f"{box_data['class_id']} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
    return final_image, yolo_annotations

def generate_multi_instance_dataset_for_all_items(num_images_per_item: int, start_image_count: int):
    """
    為每個物品類別生成多張包含該物品2-4個實例的圖像。

    :param num_images_per_item: 要為每個物品類別生成的圖像數量。
    :param start_image_count: 圖像文件命名的起始計數。
    :return: 更新後的圖像計數。
    """
    global items, class_to_id # 確保可以訪問全域物品列表和ID映射
    print(f"開始為所有物品生成多實例圖片集，每個物品生成 {num_images_per_item} 張...")
    
    current_image_count = start_image_count
    total_generated_for_all_items = 0

    if not items:
        print("錯誤：物品列表 (items) 為空，無法生成多實例數據集。")
        return start_image_count

    for item_class_name, _ in items: # 我們只需要 class_name
        print(f"正在為物品 '{item_class_name}' 生成多實例圖片...")
        generated_for_this_item = 0
        for i in range(num_images_per_item):
            # 隨機決定當前圖片中該物品的實例數量 (2到4個)
            num_instances_for_this_image = random.randint(2, 4)
            
            # 隨機決定最大重疊率
            # 可以根據需要調整這個範圍，例如更傾向於低重疊或高重疊
            max_overlap_for_this_image = random.uniform(0.1, 0.6) 

            # 呼叫先前定義的函式生成單張圖片
            image, annotations = generate_same_item_multiple_instances(
                item_class_name=item_class_name,
                num_instances=num_instances_for_this_image,
                max_overlap_val=max_overlap_for_this_image
            )

            if image is not None and annotations:
                # 保存圖片
                image_filename = f'../assets/dataset/images/image_{current_image_count:04d}.png'
                cv2.imwrite(image_filename, image)
                
                # 保存標註
                label_filename = f'../assets/dataset/labels/image_{current_image_count:04d}.txt'
                with open(label_filename, 'w') as f:
                    f.write('\n'.join(annotations))
                
                # print(f"  已生成圖片: {image_filename} 包含 {num_instances_for_this_image} 個 '{item_class_name}'")
                current_image_count += 1
                generated_for_this_item += 1
                total_generated_for_all_items += 1
            else:
                print(f"  警告：為 '{item_class_name}' 生成多實例圖片失敗 (第 {i+1}/{num_images_per_item} 張嘗試)。")
        
        print(f"物品 '{item_class_name}' 完成，共生成 {generated_for_this_item} 張多實例圖片。")

    print(f"所有物品的多實例圖片生成完成。總共生成了 {total_generated_for_all_items} 張新圖片。")
    print(f"最終的圖片計數為: {current_image_count}")
    return current_image_count

last_image_count = 0
total_images_to_generate = 160
overlapped_images_generated = 0
min_overlap_percentage = 0 # 要求至少30%的圖片有重疊
force_overlap_max_attempts_outer = 500 # 為強制重疊圖片設定的外部嘗試次數

for i in range(total_images_to_generate):
    should_force_overlap_for_this_image = False
    # 如果剩餘需要生成的重疊圖片數量大於等於剩餘總圖片數量，則強制這張圖片重疊
    if (total_images_to_generate * min_overlap_percentage - overlapped_images_generated) >= (total_images_to_generate - i):
        should_force_overlap_for_this_image = True

    image_generated_successfully = False
    attempt_for_current_image = 0
    
    while not image_generated_successfully and attempt_for_current_image < (force_overlap_max_attempts_outer if should_force_overlap_for_this_image else 1):
        image, annotations, has_overlap = generate_image(force_this_image_to_overlap=should_force_overlap_for_this_image)
        
        if should_force_overlap_for_this_image:
            if has_overlap:
                image_generated_successfully = True
            else: # 強制重疊但沒成功，計次並可能重試
                print(f"警告: 強制重疊圖像生成嘗試失敗 image_{last_image_count + i:04d} (Attempt: {attempt_for_current_image + 1})")
        else: # 非強制重疊，生成一次即可
            image_generated_successfully = True
        
        attempt_for_current_image += 1

    if not image_generated_successfully: # 多次嘗試後仍無法生成所需的重疊圖片
        print(f"錯誤: 無法生成 {'強制重疊' if should_force_overlap_for_this_image else ''} 圖片 image_{last_image_count + i:04d}.png，將使用最後一次嘗試的結果。")
        # 即使不符合強制重疊要求，也繼續使用最後一次生成的圖片

    if has_overlap:
        overlapped_images_generated += 1

    # 保存圖片
    cv2.imwrite(f'../assets/dataset/images/image_{last_image_count + i:04d}.png', image)
    # 保存標註
    with open(f'../assets/dataset/labels/image_{last_image_count + i:04d}.txt', 'w') as f:
        f.write('\n'.join(annotations))
    
    if (i + 1) % 100 == 0:
        print(f"已生成 {i+1}/{total_images_to_generate} 張圖片. 目前重疊比例: {overlapped_images_generated / (i+1):.2f}")

    # 決定每個物品要生成多少張「多實例」的圖片
num_multi_instance_images_per_item = 250 # 例如，為每個物品生成10張

    # 呼叫新的函式
last_image_count = generate_multi_instance_dataset_for_all_items(
num_images_per_item=num_multi_instance_images_per_item,
start_image_count=total_images_to_generate
)

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