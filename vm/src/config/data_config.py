from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

class ItemType(Enum):
    # Treasure items
    crystal = 0
    diamond = auto()
    emerald = auto()

    # Landmark items
    coin = auto()
    compass = auto()
    coral = auto()
    fossil = auto()
    key = auto()
    letter = auto()
    shell = auto()
    treasure_box = auto()

class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

@dataclass
class DataConfig:
    item_list: list[ItemType]

    # Image control
    image_size: tuple[int, int] = (160, 160)
    split: DatasetSplit = DatasetSplit.TRAIN

    # Item control
    item_scale_range: tuple[float, float] = (0.2, 0.4) # As a fraction of image size
    item_rotate_range: tuple[int, int] = (0, 360)
    item_blur_kernel_range: tuple[int] = (1, 3, 5)
    item_blur_kernel: Optional[int] = None
    max_random_placement_trials: int = 50

    # Position control
    force_overlap: bool = False
    max_overlap: float = 0.3

    # Augmentation
    apply_shear: bool = True
    shear_range: tuple[float, float] = (-0.05, 0.05)
    shear: Optional[float] = None
    brightness_range: tuple[float, float] = (0.9, 1.2)
    brightness: Optional[float] = None
    contrast_range: tuple[float, float] = (0.8, 1)
    contrast: Optional[float] = None

    # Reproducibility
    seed: Optional[int] = None

class Data:
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 item_list: Optional[list[ItemType]] = None,
                 bboxes: Optional[list[tuple[float, float, float, float]]] = None,
                 annotations: Optional[list[tuple[ItemType, float, float, float, float]]] = None,
                 config: Optional[DataConfig] = None) -> None:
        self.image = image
        self.item_list = item_list
        self.bboxes = bboxes
        self.annotations = annotations
        self.config = config

    def to_string(self) -> str:
        return f"Data(image_shape={self.image.shape if self.image is not None else None}, " \
               f"item_list={self.item_list}, " \
               f"bboxes={self.bboxes}, " \
               f"annotations={self.annotations}), " \
               f"config={self.config})"

    def plot(self) -> None:
        if self.image is None:
            raise ValueError("Image data is not available for plotting.")

        image = Image.fromarray(self.image.astype('uint8'))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        draw.rectangle([0, 0, image.width-1, image.height-1], outline="black", width=3)

        # Plot annotations
        for annotation in self.annotations or []:
            item_type, x_min, y_min, x_max, y_max = annotation
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min), item_type.name, fill="red", font=font)
        
        # Display the image
        width, height = image.size
        dpi = 100
        figsize = width / dpi, height / dpi

        fig = plt.figure(figsize=figsize, dpi=dpi)

        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()