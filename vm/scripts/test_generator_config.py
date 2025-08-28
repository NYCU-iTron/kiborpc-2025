from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.data_config import Data, ItemType
from config.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()
    generator = DataGenerator()

    config = GeneratorConfig(
        item_num_range=(3, 3),
        item_types=[ItemType.coin],
        force_overlap=False,
        max_overlap=0.2,
        apply_full_aug=False
    )
