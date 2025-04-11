import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from pathlib import Path

INPUT_DIR = "../assets/item_template_images"
OUTPUT_DIR = "../assets/augmented"

# === 建立輸出資料夾 ===
# Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
# Path(f"{OUTPUT_DIR}/images").mkdir(exist_ok=True)
# Path(f"{OUTPUT_DIR}/labels").mkdir(exist_ok=True)

def remove_outer_white_to_transparent(image: np.ndarray) -> np.ndarray:
    # 灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 先找出白色區域
    _, white_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # flood fill 找到與邊界相連的白色區塊（背景）
    h, w = gray.shape
    floodfilled = white_mask.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfilled, mask, (0, 0), 128)  # 將外圍白區填成128

    # 建立 alpha 遮罩：將128區域（背景）設為透明，其它保留
    alpha = np.where(floodfilled == 128, 0, 255).astype(np.uint8)

    # 合併 alpha 通道
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, alpha])

    return rgba

def show_rgba_on_checkerboard(rgba: np.ndarray, tile_size: int = 20) -> None:
    h, w = rgba.shape[:2]

    # 產生棋盤格背景
    num_rows = h // tile_size + 1
    num_cols = w // tile_size + 1
    checkerboard = np.zeros((num_rows * tile_size, num_cols * tile_size, 3), dtype=np.uint8)

    for y in range(num_rows):
        for x in range(num_cols):
            if (x + y) % 2 == 0:
                checkerboard[y * tile_size:(y + 1) * tile_size,
                                x * tile_size:(x + 1) * tile_size] = 200  # 淺灰
            else:
                checkerboard[y * tile_size:(y + 1) * tile_size,
                                x * tile_size:(x + 1) * tile_size] = 255  # 白色

    # 裁切背景大小符合圖片
    checkerboard = checkerboard[:h, :w]

    # 取得 alpha 通道，並進行正規化
    alpha = rgba[:, :, 3] / 255.0
    alpha_3c = np.stack([alpha]*3, axis=-1)

    # 混合原圖與背景
    rgb = rgba[:, :, :3]
    blended = (rgb * alpha_3c + checkerboard * (1 - alpha_3c)).astype(np.uint8)

    # 顯示結果
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

raw_images = list(Path(INPUT_DIR).glob("*.png"))
image_data = {
    "image": [],
    "class": []
}

image_datas = []
for image_path in raw_images:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    rgba = remove_outer_white_to_transparent(img)
    # show_rgba_on_checkerboard(rgba)
    class_name = image_path.stem

    image_data['image'] = rgba
    image_data['class'] = class_name
    image_datas.append(image_data)


# print(len(images))

composite_data = {
    "image": np.ones((500, 800, 3), dtype=np.uint8) * 255,  # 白色 canvas
    "objects": []
}

x, y = np.random.randint(0, 700), np.random.randint(0, 300)
rotation = np.random.choice([0, 90, 180, 270])
flipped = np.random.choice([True, False])

bbox = [x / 800, y / 500, image_data[0]['image'].shape[1] / 800,  image_data[0]['image'].shape[0] / 500]
composite_data["objects"].append({
    "class_id":  image_data[0]['class'],
    "bbox": bbox,
    "rotation": rotation,
    "flipped": flipped,
})