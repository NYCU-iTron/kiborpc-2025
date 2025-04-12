import os
import cv2
import numpy as np
from PIL import Image
import random
import math

# 創建輸出目錄
os.makedirs('dataset/images', exist_ok=True)
os.makedirs('dataset/labels', exist_ok=True)

# 讀取所有物品圖片
items = []
class_names = []
for item in os.listdir('.'):
    if item.endswith('.png'):
        # 使用PIL讀取圖片以保持透明度
        pil_img = Image.open(item)
        # 確保圖片是RGBA模式
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        # 轉換為numpy數組
        img = np.array(pil_img)
        # 使用原始檔名（不含副檔名）作為class name
        class_name = os.path.splitext(item)[0]
        items.append((class_name, img))
        class_names.append(class_name)

# 設置輸出圖片大小
output_size = (720, 720)

def rotate_image(image, angle):
    # 獲取圖片中心點
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 創建旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 執行旋轉
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    return rotated

def generate_image():
    # 創建白色背景（RGBA）
    background = np.full((output_size[0], output_size[1], 4), 255, dtype=np.uint8)
    
    # 隨機選擇2-4個物品
    num_items = random.randint(2, 4)
    selected_items = random.sample(items, num_items)
    
    annotations = []
    
    for class_name, item_img in selected_items:
        # 隨機縮放
        scale = random.uniform(0.2, 0.4)
        new_size = (int(item_img.shape[1] * scale), int(item_img.shape[0] * scale))
        resized = cv2.resize(item_img, new_size)
        
        # 隨機旋轉
        angle = random.uniform(0, 360)
        rotated = rotate_image(resized, angle)
        
        # 隨機位置
        x = random.randint(0, output_size[0] - new_size[0])
        y = random.randint(0, output_size[1] - new_size[1])
        
        # 將物品放置到背景上
        alpha_mask = rotated[:, :, 3] > 0
        for c in range(3):  # 只處理RGB通道
            background[y:y+new_size[1], x:x+new_size[0], c][alpha_mask] = rotated[:, :, c][alpha_mask]
        # 處理alpha通道
        background[y:y+new_size[1], x:x+new_size[0], 3][alpha_mask] = rotated[:, :, 3][alpha_mask]
        
        # 生成YOLO格式的標註
        # YOLO格式：<class> <x_center> <y_center> <width> <height>
        x_center = (x + new_size[0]/2) / output_size[0]
        y_center = (y + new_size[1]/2) / output_size[1]
        width = new_size[0] / output_size[0]
        height = new_size[1] / output_size[1]
        
        # 使用class name而不是數字索引
        annotations.append(f"{class_name} {x_center} {y_center} {width} {height}")
    
    return background, annotations

# 生成100張圖片
for i in range(100):
    image, annotations = generate_image()
    
    # 保存圖片
    cv2.imwrite(f'dataset/images/image_{i:04d}.png', image)
    
    # 保存標註
    with open(f'dataset/labels/image_{i:04d}.txt', 'w') as f:
        f.write('\n'.join(annotations))

# 生成classes.txt，使用原始檔名
with open('dataset/classes.txt', 'w') as f:
    for class_name in class_names:
        f.write(f'{class_name}\n')

print("數據集生成完成！") 