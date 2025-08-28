from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

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

@dataclass
class DataConfig:
    item_list: list[ItemType]

    # Image control
    image_size: tuple[int, int] = (160, 160)

    # Item control
    item_scale_range: tuple[float, float] = (0.05, 0.15)
    item_rotate_range: tuple[int, int] = (0, 360)
    item_blur_kernel_range: tuple[int] = (1, 3, 5)
    max_random_placement_trials: int = 50

    # Position control
    force_overlap: bool = False
    max_overlap: float = 0.3

    # Augmentation
    apply_full_aug: bool = True
    shear_range: tuple[float, float] = (-0.05, 0.05)
    brightness_range: tuple[float, float] = (0.9, 1.3)
    contrast_range: tuple[float, float] = (0.9, 1.1)

    # Reproducibility
    seed: Optional[int] = None

class Data:
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 item_list: Optional[list[ItemType]] = None,
                 bboxes: Optional[list[tuple[float, float, float, float]]] = None,
                 annotations: Optional[list[tuple[ItemType, float, float, float, float]]] = None) -> None:
        self.image = image
        self.item_list = item_list
        self.bboxes = bboxes
        self.annotations = annotations

    def to_string(self) -> str:
        return f"Data(image_shape={self.image.shape if self.image is not None else None}, " \
               f"item_list={self.item_list}, " \
               f"bboxes={self.bboxes}, " \
               f"annotations={self.annotations})"

    def plot(self) -> None:
        if self.image is None:
            raise ValueError("Image data is not available for plotting.")
        
        plt.imshow(self.image)
        plt.axis('off')
        
        if self.bboxes and self.item_list:
            for bbox, item in zip(self.bboxes, self.item_list):
                x_min, y_min, x_max, y_max = bbox
                width, height = x_max - x_min, y_max - y_min
                rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x_min, y_min - 10, item.name, color='red', fontsize=12, weight='bold')
        
        plt.show()