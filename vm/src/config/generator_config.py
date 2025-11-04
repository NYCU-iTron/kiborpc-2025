from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GeneratorConfig:
    seed: Optional[int] = None # Reproducibility
    generate_images: bool = True

    # Single item images
    single_item_images_num: int = 10 # per landmark item
    single_item_num_choices: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    single_item_num_weights: list[int] = field(default_factory=lambda: [4, 4, 4, 1])
    single_force_overlap_ratio: float = 0.7
    single_force_max_overlap: float = 0.6

    # Multi item images
    multi_item_images_num: int = 20
    treasure_item_ratio: float = 0.5
    multi_item_num_choices: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    multi_item_num_weights: list[int] = field(default_factory=lambda: [2, 4, 4, 4])
    item_type_num_choices: list[int] = field(default_factory=lambda: [2, 3]) # Should be at least 2
    item_type_num_wieights: list[int] = field(default_factory=lambda: [8, 2])
    multi_force_overlap_ratio: float = 0.4
    multi_force_max_overlap: float = 0.3

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