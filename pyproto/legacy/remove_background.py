import os
import cv2
import numpy as np
from PIL import Image

def remove_background(image_path, output_path):
    # 使用OpenCV讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # 轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用閾值處理找出非白色區域
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 使用形態學操作來填充物件內部的孔洞
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找出最大的輪廓（假設是主要物件）
        max_contour = max(contours, key=cv2.contourArea)
        
        # 創建遮罩
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        
        # 使用形態學運算來平滑邊緣
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 合併alpha通道
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask])
    else:
        # 如果沒有找到輪廓，保持原圖不變
        b, g, r = cv2.split(img)
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
        rgba = cv2.merge([b, g, r, alpha])
    
    # 使用PIL保存為PNG以保持透明度
    pil_img = Image.fromarray(rgba)
    pil_img.save(output_path, 'PNG')

def main():
    # 創建輸出目錄
    output_dir = '../assets/item_template_images/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理所有PNG文件
    input_dir = '../assets/item_template_images'
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f'處理圖片: {filename}')
            remove_background(input_path, output_path)
    
    print('所有圖片處理完成！')

if __name__ == '__main__':
    main() 