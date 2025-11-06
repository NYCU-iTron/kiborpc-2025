import logging
from data_generator import DataGenerator
from config.generator_config import GeneratorConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    config = GeneratorConfig(
        type1_images_num_per_landmark=7, # total = 7 * 8 = 56
        type2_images_num_per_treasure=20, # total = 20 * 3 = 60
        type3_item_images_num_per_treasure=20, # total = 20 * 3 = 60
        generate_images=True,
    )

    generator.generate_dataset(config)
