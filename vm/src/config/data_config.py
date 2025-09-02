from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class ItemType(Enum):
    # Treasure items
    crystal = 0 # Important: the first item id should be 0 for yolo format
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
    train = auto()
    valid = auto()

@dataclass
class DataConfig:
    item_list: list[ItemType]

    # Image control
    image_size: tuple[int, int] = (160, 160)

    # Item control
    item_scale_range: tuple[float, float] = (0.2, 0.4) # As a fraction of image size
    item_rotate_range: tuple[int, int] = (0, 360)
    item_blur_kernel_range: tuple[int] = (1, 3, 5)
    item_blur_kernel: Optional[int] = None
    max_random_placement_trials: int = None # Depends on whether force_overlap is True or False

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

    def __post_init__(self):
        if self.max_random_placement_trials is None:
            if self.force_overlap:
                self.max_random_placement_trials = 100
            else:
                self.max_random_placement_trials = 50

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
        self.split: Optional[DatasetSplit] = None

    def to_string(self) -> str:
        return f"Data(image_shape={self.image.shape if self.image is not None else None}, " \
               f"item_list={self.item_list}, " \
               f"bboxes={self.bboxes}, " \
               f"annotations={self.annotations}), " \
               f"config={self.config})"
    
    def plot(self) -> None:
        if self.image is None:
            raise ValueError("Image data is not available for plotting.")

        height, width, _ = self.image.shape
        
        # Set DPI and calculate figure size to match image pixels 1:1
        dpi = 100
        figsize = width / float(dpi), height / float(dpi)

        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
        
        # Display the image without scaling
        ax.imshow(self.image)

        # Remove padding and axes for a clean display
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')

        # Draw a border around the entire image
        ax.add_patch(patches.Rectangle((0, 0), width-1, height-1, linewidth=3, edgecolor='black', facecolor='none'))

        # Plot annotations using matplotlib patches
        for item, bbox in zip(self.item_list, self.bboxes):
            x_min, y_min, x_max, y_max = bbox
            
            # Create a Rectangle patch for the bounding box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add a text label
            ax.text(
                x_min, y_min, item.name,
                color='white', verticalalignment='top',
                bbox=dict(facecolor='red', alpha=0.7, pad=1)
            )
        
        plt.show()