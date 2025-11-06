from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GeneratorConfig:
    seed: Optional[int] = None # Reproducibility
    generate_images: bool = True

    # Type 1: Contain only one kind of landmark item
    type1_images_num_per_landmark: int = 10 # per landmark item, total = type1_images_num * 8
    type1_landmark_num_choices: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    type1_landmark_num_weights: list[int] = field(default_factory=lambda: [4, 4, 4, 1])
    type1_force_overlap_ratio: float = 0.7
    type1_force_max_overlap: float = 0.6

    # Type 2: Contain one kind of treasure item and one kind of landmark item
    type2_images_num_per_treasure: int = 15 # per treasure item, total = type2_images_num * 3
    type2_landmark_num_choices: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    type2_landmark_num_weights: list[int] = field(default_factory=lambda: [4, 4, 2, 1])

    # Type 3: Contain three items: one treasure item and two different landmark items
    type3_item_images_num_per_treasure: int = 20 # per treasure item, total = type3_item_images_num * 3

    # Dataset split
    train_ratio: float = 0.8
    valid_ratio: float = 0.2
    max_valid: int = 4000

    def __post_init__(self):
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if not (0 < self.valid_ratio < 1):
            raise ValueError("valid_ratio must be between 0 and 1")
        if not (self.train_ratio + self.valid_ratio == 1):
            raise ValueError("train_ratio + valid_ratio must be equal to 1")
