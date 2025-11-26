from pathlib import Path
import cv2
import numpy as np

def get_brightness(image: Path) -> float:
    """
    Calculate the brightness of an image.

    Args:
        image (Path): The path to the image file.

    Returns:
        float: The brightness value of the image.
    """
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Cannot read image: {image}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate brightness as the mean of pixel values
    mean_brightness = gray.mean()
    print(f"Mean brightness: {mean_brightness:.2f}")

    overexposed_ratio = np.sum(gray > 240) / gray.size
    print(f"Overexposed pixel ratio: {overexposed_ratio:.4f}")

    if mean_brightness > 200 or overexposed_ratio > 0.05:
        print("圖片過亮，建議調整曝光")
    elif mean_brightness < 50:
        print("圖片偏暗，可能影響辨識")
    else:
        print("亮度正常")

    return

base_dir = Path(__file__).resolve().parent
test_set_dir = (base_dir / '../assets/test_set').resolve()
get_brightness(test_set_dir / '61.png')
