from dataclasses import dataclass
from typing import Optional
from .data_config import ItemType 

@dataclass
class GeneratorConfig:
    image_size: tuple[int, int] = (160, 160)

    # Item control
    item_types: list[ItemType]
    item_num_range: tuple[int, int] = (1, 5)
    item_scale_range: tuple[float, float] = (0.05, 0.15)
    item_rotate_range: tuple[int, int] = (0, 360)
    item_blur_kernel_range: list[int] = (1, 3, 5)
    max_random_placement_trials: int = 50

    # Position control
    force_overlap: bool = False
    max_overlap: float = 0.3

    # Augmentation
    apply_full_aug: bool = True

    # Reproducibility
    seed: Optional[int] = None