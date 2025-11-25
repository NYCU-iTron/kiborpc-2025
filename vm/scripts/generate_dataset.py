import logging
from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    total_images = 150000
    type1_ratio = 0.3  # only one kind of landmark item
    type2_ratio = 0.4  # one treasure item + repeated one kind of landmark item
    type3_ratio = 0.3  # one treasure item + two different landmark items

    config = GeneratorConfig(
        type1_images_num_per_landmark=int(total_images * type1_ratio / 8),
        type2_images_num_per_treasure=int(total_images * type2_ratio / 3),
        type3_item_images_num_per_treasure=int(total_images * type3_ratio / 3),
        seed=42,
    )

    generator.generate_dataset(config)
