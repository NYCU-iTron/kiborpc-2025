import os
import cv2
import numpy as np
from PIL import Image

def remove_background(image_path, output_path):
    # 使用PIL讀取圖片
    pil_img = Image.open(image_path)
    # 確保圖片是RGBA模式
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    
    # 獲取圖片數據
    data = pil_img.getdata()
    
    # 創建新的圖片數據
    new_data = []
    for item in data:
        # 如果像素是白色（RGB值都接近255），則設為透明
        if item[0] > 250 and item[1] > 250 and item[2] > 250:
            new_data.append((0, 0, 0, 0))  # 完全透明
        else:
            new_data.append(item)  # 保持原樣
    
    # 創建新圖片
    new_img = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    new_img.putdata(new_data)
    
    # 保存結果
    new_img.save(output_path, 'PNG')

def main():
    # 創建輸出目錄
    output_dir = 'processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理所有PNG文件
    for filename in os.listdir('.'):
        if filename.endswith('.png'):
            input_path = filename
            output_path = os.path.join(output_dir, filename)
            print(f'處理圖片: {filename}')
            remove_background(input_path, output_path)
    
    print('所有圖片處理完成！')

if __name__ == '__main__':
    main() 