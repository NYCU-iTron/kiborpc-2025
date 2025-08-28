import logging
from data_generator import DataGenerator
from config.data_config import ItemType, DataConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    setup_logging()
    generator = DataGenerator()

    config = DataConfig(
        item_list=[ItemType.compass] * 4,
        image_size=(200, 200),
        force_overlap=False,
        apply_shear=False,
        max_overlap=0.3,
    )

    data = generator.generate_single_data(config)
    logger.info(data.to_string())
    data.plot()
