from data_generator import DataGenerator

data_generator = DataGenerator()
data_generator.generate_yaml()
data_generator.generate_data(10)
data_generator.generate_debug_images()
data_generator.split_data()
