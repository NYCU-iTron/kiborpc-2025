import logging
from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    config = GeneratorConfig(
        single_item_images_num=1500, # each item, times 8 get total
        multi_item_images_num=78000,
        seed=42,
    )

    generator.generate_dataset(config)
