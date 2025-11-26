import logging
from data_generator import DataGenerator
from config.data_config import ItemType, DataConfig
from config.logging_config import setup_logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    setup_logging()

    generator = DataGenerator()
    config = DataConfig(
        item_list=[ItemType.shell, ItemType.diamond, ItemType.diamond],
        image_shape=(150, 150),
        force_overlap=False,
        seed=42,
    )

    data = generator.generate_single_data(config)
    logger.info(data.to_string())
    data.plot()
