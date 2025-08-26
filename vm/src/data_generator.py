from pathlib import Path
import logging
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import random
from data import Data, ItemType

class DataGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Get item images directory
        self.item_image_dir = Path(__file__).parent.parent / "item_images"
        if not self.item_image_dir.is_dir():
            raise FileNotFoundError(f"Item image directory not found: {self.item_image_dir}")
        
        # Load item images, accept only webp
        self.item_dict = {}
        for item_type in ItemType:
            item_image_file = self.item_image_dir / f"{item_type.name}.webp"
            if not item_image_file.is_file():
                raise FileNotFoundError(f"Item image not found: {item_image_file}")
        
            item_image = Image.open(item_image_file).convert("RGBA")
            self.item_dict[item_type] = np.array(item_image)
            self.logger.info(f"Loaded item image: name='{item_type.name}.webp', id={item_type.value}")

        # Prepare dataset directories
        self.dataset_dir = Path(__file__).parent.parent / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Generate classes.txt
        classes_file = self.dataset_dir / "classes.txt"
        with open(classes_file, "w") as f:
            for item_type in ItemType:
                f.write(f"{item_type.value} {item_type.name}\n")

    def generate_single_data(self, max_overlap: float, is_force_overlapped: bool, overlapping_num: int, max_attempts: int = 50) -> Data:
        if max_overlap < 0 or max_overlap > 1:
            raise ValueError("max_overlap must be between 0 and 1")
        
        if max_overlap is None:
            max_overlap = random.uniform(0.5, 0.8)
        
        if is_force_overlapped:
            max_overlap = random.uniform(0.05, 0.25) 
        
        if overlapping_num is None:
            overlapping_num = random.randint(2, 5)

        if overlapping_num is not None:
            place_item_num = min(overlapping_num, len(self.item_dict))
        else:
            place_item_num = min(random.choice([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5]), len(self.item_dict))
    
        if is_force_overlapped and place_item_num < 2 and len(self.item_dict) >= 2:
            place_item_num = 2

        item_type_pool = [item_type for item_type in self.item_dict.keys()] * 5
        selected_item_types = random.choices(item_type_pool, k=place_item_num)

        # Generate plane image
        output_size = (640, 640)
        image = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

        for item_type in selected_item_types:
            # Randomly scale item image
            scale = random.uniform(0.05, 0.15)
            item_image = self.item_dict[item_type]
            new_size = (int(item_image.shape[1] * scale), int(item_image.shape[0] * scale))
            item_image = cv2.resize(item_image, new_size, interpolation=cv2.INTER_CUBIC)

            # Remove transparency noise
            alpha = item_image[:, :, 3]
            _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            item_image[:, :, 3] = alpha

            # Randomly rotate item image
            angle = random.uniform(0, 360)
            item_image = self.rotate_image(item_image, angle)
            height, width = item_image.shape[:2]

            # Item-specific blur is still applied before placing on background
            item_blur_kernel = random.choice(list(range(1, 7, 2))) 
            item_image[:, :, :3] = cv2.GaussianBlur(item_image[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)
        
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        # Compute the center of the image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Compute the bounding box of the rotated image
        cos = np.abs(np.cos(np.radians(angle)))
        sin = np.abs(np.sin(np.radians(angle)))
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate the image
        image = cv2.warpAffine(image, 
                               rotation_matrix, 
                               (new_width, new_height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
        
        # Remove transparency noise
        alpha = image[:, :, 3]
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        image[:, :, 3] = alpha
        
        # Find the bounding box of the non-transparent area
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop the image to the bounding box
        image = image[rmin:rmax+1, cmin:cmax+1]
        
        return image