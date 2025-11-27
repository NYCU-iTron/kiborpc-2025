from pathlib import Path
import cv2
import numpy as np

def analyze_image_quality(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Image not found: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 核心指標
    mean_brightness = gray.mean()
    min_val, _, _, _ = cv2.minMaxLoc(gray)

    print(f"--- {image_path.name} ---")
    print(f"平均亮度: {mean_brightness:.1f}, 最暗點: {min_val}")

    is_overexposed = False

    # 簡化後的判斷邏輯：

    # 1. 極度過曝 (整張圖幾乎全白)
    if mean_brightness > 250:
        print("❌ 過曝: 整體太亮 (Mean > 250)")
        is_overexposed = True

    # 2. 黑色細節遺失 (最暗的地方都已經是亮灰/白色)
    #    同時確保不是因為圖案太小導致的誤判 (所以加一個 mean > 230 的輔助檢查)
    elif min_val > 80 and mean_brightness > 230:
        print(f"❌ 過曝: 黑色細節遺失 (Min {min_val} > 80 且整體偏亮)")
        is_overexposed = True

    if not is_overexposed:
        print("✅ 正常")

# --- 測試執行 ---
base_dir = Path(__file__).resolve().parent
test_set_dir = (base_dir / '../assets/test_set').resolve()

test_images = ['54.png', '55.png', '56.png', '57.png', '58.png', '63.png',
               '59.png', '60.png', '61.png', '62.png',  '64.png', '65.png']

for img_name in test_images:
    analyze_image_quality(test_set_dir / img_name)
