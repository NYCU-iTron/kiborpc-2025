from data_generator import DataGenerator
from config.data_config import DataConfig
from config.data_config import ItemType
from config.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()
    generator = DataGenerator()

    config = DataConfig(
        item_list=[ItemType.coin] * 3,
    )

    data = generator.generate_single_data(config)
    data.plot()
