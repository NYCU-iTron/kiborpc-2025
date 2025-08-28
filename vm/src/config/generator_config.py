from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    # Single item images
    single_item_images_num: int = 10 # per item
    single_item_num_range: tuple[int, int] = (3, 6)
    single_force_overlap_ratio: float = 0.6

    # Multi item images
    multi_item_images_num: int = 20
    multi_item_num_range: tuple[int, int] = (2, 7)
    multi_force_overlap_ratio: float = 0.3

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