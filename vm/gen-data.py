from data_generator import DataGenerator

data_generator = DataGenerator()
data_generator.generate_yaml()
data_generator.generate_data(20000)
data_generator.split_data()
