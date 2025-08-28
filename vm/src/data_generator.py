from pathlib import Path
import logging
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import random
from config.data_config import Data, ItemType
from config.generator_config import GeneratorConfig

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

    def random_config(self) -> GeneratorConfig:
        pass
    
    def generate_single_data(self, config: GeneratorConfig) -> Data:
        # Generate plane image
        image_size = config.image_size
        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

        bboxes = []
        annotations = []
        for item_type in config.item_types:
            # Randomly scale item image
            scale = random.uniform(config.item_scale_range[0], config.item_scale_range[1])
            item_image = self.item_dict[item_type]
            item_image = self.scale_image(item_image, scale)

            # Randomly rotate item image
            angle = random.uniform(config.item_rotate_range[0], config.item_rotate_range[1])
            item_image = self.rotate_image(item_image, angle)

            # Item-specific blur is still applied before placing on background
            item_blur_kernel = random.choice(config.item_blur_kernel_range) 
            item_image[:, :, :3] = cv2.GaussianBlur(item_image[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)

            # Randomly place item image on background
            item_height, item_width = item_image.shape[:2]
            success = False
            for trial in range(config.max_random_placement_trials):
                x = random.randint(0, max(0, config.image_size[0] - item_width))
                y = random.randint(0, max(0, config.image_size[1] - item_height))
                new_bbox_coords = (x, y, x + item_width, y + item_height)

                # Make sure the item is within the image bounds
                if not (x >= 0 and y >= 0 and x + item_width <= config.image_size[0] and y + item_height <= config.image_size[1]):
                    continue

                # Check for overlap with existing items
                if len(bboxes) > 0:
                    ious = [self.calc_bbox_iou(new_bbox_coords, bbox) for bbox in bboxes]
                    max_iou = max(ious)

                    if max_iou > config.max_overlap:
                        continue

                    if config.force_overlap and max_iou == 0:
                        continue

                bboxes.append(new_bbox_coords)
                annotations.append((item_type, x, y, x + item_width, y + item_height))
                
                # Place item image on background using alpha channel as mask
                alpha_mask = item_image[:, :, 3] > 0
                for c in range(3): # RGB channels
                    image[y : y + item_height, x : x + item_width, c][alpha_mask] = item_image[:, :, c][alpha_mask]
                
                success = True

            if not success:
                self.logger.warning(f"Failed to place item '{item_type.name}' after {config.max_random_placement_trials} trials.")
                continue

    # ------------------------------ Tool Functions ------------------------------ #
    def scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        # Remove transparency noise
        alpha = image[:, :, 3]
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        image[:, :, 3] = alpha

        return image

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

    def calc_bbox_iou(box1, box2) -> float:
        # box = (x1, y1, x2, y2)
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        # Calculate intersection
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        
        return inter_area / union_area