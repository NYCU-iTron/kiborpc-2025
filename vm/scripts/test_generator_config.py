import logging
from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    config = GeneratorConfig(
        single_item_images_num=0,
        multi_item_images_num=10000,
        generate_images=False,
    )

    generator.generate_dataset(config)