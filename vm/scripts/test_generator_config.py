import logging
from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    config = GeneratorConfig(
        single_item_images_num=1,
        multi_item_images_num=5,
    )

    generator.generate_dataset(config)