from pathlib import Path
import logging
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import random
from config.data_config import Data, DataConfig, ItemType, DatasetSplit
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

        for split in DatasetSplit:
            (self.images_dir / split.name).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split.name).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Prepared dataset directories under: {self.dataset_dir}")

        # Generate yaml file
        yaml_file = self.dataset_dir / "data.yaml"
        with open(yaml_file, "w") as f:
            f.write(
                f"path: {self.dataset_dir.resolve()}\n"
                f"train: images/train\n"
                f"val: images/val\n\n"
                f"names:\n"
            )
            for item_type in ItemType:
                f.write(f"  {item_type.value}: {item_type.name}\n")

        self.logger.info(f"Generated yaml file: {yaml_file}")

    # ------------------------------ Main Functions ------------------------------ #
    def generate_single_data(self, config: DataConfig) -> Data:
        # Generate plane image
        image_size = config.image_size
        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

        bboxes = []
        for item in config.item_list:
            # Randomly scale item image
            scale = random.uniform(config.item_scale_range[0], config.item_scale_range[1])
            item_image = self.item_dict[item].copy()
            target_width = int(config.image_size[0] * scale)
            scale = target_width / item_image.shape[1]
            item_image = self.scale_image(item_image, scale)

            # Randomly rotate item image
            angle = random.uniform(config.item_rotate_range[0], config.item_rotate_range[1])
            item_image = self.rotate_image(item_image, angle)

            # Blur item image
            if config.item_blur_kernel:
                item_blur_kernel = config.item_blur_kernel
            else:
                item_blur_kernel = random.choice(config.item_blur_kernel_range) 
                config.item_blur_kernel = item_blur_kernel

            item_image[:, :, :3] = cv2.GaussianBlur(item_image[:, :, :3], (item_blur_kernel, item_blur_kernel), 0)

            # Randomly place item image on background
            item_height, item_width = item_image.shape[:2]
            success = False
            for trial in range(config.max_random_placement_trials):
                x = random.randint(0, max(0, config.image_size[0] - item_width))
                y = random.randint(0, max(0, config.image_size[1] - item_height))
                new_bbox = (x, y, x + item_width, y + item_height)

                # Make sure the item is within the image bounds
                if not (x >= 0 and y >= 0 and x + item_width <= config.image_size[0] and y + item_height <= config.image_size[1]):
                    continue

                # Check for overlap with existing items
                if len(bboxes) > 0:
                    ious = [self.calc_bbox_iou(new_bbox, bbox) for bbox in bboxes]
                    overlaps = [self.calc_bbox_max_overlap_ratio(new_bbox, bbox) for bbox in bboxes]
                    
                    max_iou = max(ious)
                    max_overlap = max(overlaps)

                    if max_iou > config.max_overlap or max_overlap > config.max_overlap:
                        continue

                    if config.force_overlap and max_iou == 0:
                        continue

                # Place item image on background using alpha channel as mask
                alpha = item_image[:, :, 3] / 255.0
                for c in range(3):  # RGB channels
                    image[y:y+item_height, x:x+item_width, c] = (
                        alpha * item_image[:, :, c] +
                        (1 - alpha) * image[y:y+item_height, x:x+item_width, c]
                    ).astype(np.uint8)

                bboxes.append(new_bbox)
                success = True
                break

            if not success:
                self.logger.warning(f"Failed to place item '{item.name}' after {config.max_random_placement_trials} trials.")
                continue
        
        if len(bboxes) == 0:
            self.logger.warning("No items placed on the image.")
            return None
        
        # Apply shear operation
        if config.apply_shear:
            if config.shear:
                shear_factor = config.shear
            else:
                shear_factor = random.uniform(config.shear_range[0], config.shear_range[1])
                config.shear = shear_factor

            shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            
            height, width = image.shape[:2]
            is_rgba = image.shape[2] == 4
            image_rgb_part = image[:,:,:3]
            
            alpha_part_sheared = None

            # Shear alpha channel
            if is_rgba:
                alpha_part_sheared = cv2.warpAffine(image[:,:,3], shear_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Shear RGB part
            image_rgb_sheared = cv2.warpAffine(image_rgb_part, shear_matrix, (width, height))

            # Adjust Bounding Boxes for Shear
            new_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                corners = np.array([
                    [x1, y1], [x2, y1], 
                    [x1, y2], [x2, y2]])
                
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
                new_x1 = np.clip(new_x1, 0, width)
                new_y1 = np.clip(new_y1, 0, height)
                new_x2 = np.clip(new_x2, 0, width)
                new_y2 = np.clip(new_y2, 0, height)

                # Ensure x1 <= x2 and y1 <= y2
                if new_x1 > new_x2: new_x1, new_x2 = new_x2, new_x1 
                if new_y1 > new_y2: new_y1, new_y2 = new_y2, new_y1

                new_bboxes.append((new_x1, new_y1, new_x2, new_y2))

            bboxes = new_bboxes

            # Brightness/Contrast on the sheared RGB part
            new_image = image_rgb_sheared.copy()

            # Reconstruct final image
            if is_rgba and alpha_part_sheared is not None:
                image = cv2.merge((new_image, alpha_part_sheared))
            else:
                image = new_image
        
        # Apply brightness/contrast adjustment
        if config.brightness:
            brightness_factor = config.brightness
        else:
            brightness_factor = random.uniform(config.brightness_range[0], config.brightness_range[1])
            config.brightness = brightness_factor
        
        if config.contrast:
            contrast_factor = config.contrast
        else:
            contrast_factor = random.uniform(config.contrast_range[0], config.contrast_range[1])
            config.contrast = contrast_factor

        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=(brightness_factor - 1) * 128)
        
        # Prepare annotations
        annotations = []
        image_width, image_height = config.image_size
        for item, bbox in zip(config.item_list, bboxes):
            x1, y1, x2, y2 = bbox
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            item_width, item_height = x2 - x1, y2 - y1
            annotation = (item, xc/image_width, yc/image_height, item_width/image_width, item_height/image_height)
            annotations.append(annotation)
        
        # Prepare Data object
        data = Data(
            image=image,
            item_list=config.item_list,
            bboxes=bboxes,
            annotations=annotations,
            config=config
        )

        return data
    
    def generate_dataset(self, config: GeneratorConfig) -> None:
        data_config_list = []

        # Generate data configs for single item images
        for item in ItemType:
            for i in range(config.single_item_images_num):
                image_size = random.randint(config.image_size_range[0], config.image_size_range[1])
                data_config = DataConfig(
                    item_list=[item] * random.randint(config.single_item_num_range[0], config.single_item_num_range[1]),
                    image_size=(image_size, image_size),
                    force_overlap=True if random.random() < config.single_force_overlap_ratio else False,
                    max_overlap=config.single_force_overlap_ratio,
                )
                data_config_list.append(data_config)

        single_item_count = len(data_config_list)
        self.logger.info(f"Generated {single_item_count} single item image configs.")
        
        # Generate data configs for multi item images
        item_pool = list(ItemType) * 5
        for i in range(config.multi_item_images_num):
            image_size = random.randint(config.image_size_range[0], config.image_size_range[1])
            data_config = DataConfig(
                item_list=random.choices(item_pool, k=random.randint(config.multi_item_num_range[0], config.multi_item_num_range[1])),
                image_size=(image_size, image_size),
                force_overlap=True if random.random() < config.multi_force_overlap_ratio else False,
                max_overlap=config.multi_force_overlap_ratio,
            )
            data_config_list.append(data_config)
        
        multi_item_count = len(data_config_list) - single_item_count
        self.logger.info(f"Generated {multi_item_count} multi item image configs.")
        
        # Shuffle data configs
        random.shuffle(data_config_list)
        total_images = len(data_config_list)
        valid_count = min(round(total_images * config.valid_ratio), config.max_valid)
        train_count = total_images - valid_count
        self.logger.info(f"Generating dataset with {total_images} images: {train_count} train, {valid_count} valid")

        # Generate valid images
        count = 0
        for config in tqdm(data_config_list, desc="Generating dataset"):
            data = self.generate_single_data(config)
            if data is None:
                self.logger.warning("Skipping image generation due to no items placed.")
                continue

            # Determine dataset split
            if valid_count > 0:
                data.split = DatasetSplit.valid
                valid_count -= 1
            else:
                data.split = DatasetSplit.train

            # Save image
            count += 1
            self.save_data(data, count)
    
    def save_data(self, data: Data, index: int) -> None:
        if data.split is None:
            raise ValueError("Data split is not set.")
        if data.image is None:
            raise ValueError("Image data is not available for saving.")
        if data.annotations is None:
            raise ValueError("Annotations are not available for saving.")
        
        # Save image
        image_path = self.images_dir / data.split.name / f"{index:05d}.webp"
        cv2.imwrite(image_path, data.image, [cv2.IMWRITE_WEBP_QUALITY, 75])

        # Save labels
        label_path = self.labels_dir / data.split.name / f"{index:05d}.txt"
        with open(label_path, 'w') as f:
            for annotation in data.annotations:
                item, xc, yc, w, h = annotation
                f.write(f"{item.value} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # ------------------------------ Tool Functions ------------------------------ #
    def scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        
        if scale < 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC 

        image = cv2.resize(image, new_size, interpolation=interp)
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
                
        # Find the bounding box of the non-transparent area
        alpha = image[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop the image to the bounding box
        image = image[rmin:rmax+1, cmin:cmax+1]
        
        return image

    def calc_bbox_iou(self, box1, box2) -> float:
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
    
    def calc_bbox_max_overlap_ratio(self, box1, box2) -> float:
        # box = (x1, y1, x2, y2)
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        # Calculate intersection
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height
        
        # Calculate areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        if box1_area == 0 or box2_area == 0:
            return 0.0
        
        return inter_area / min(box1_area, box2_area)