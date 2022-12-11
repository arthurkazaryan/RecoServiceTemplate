import configparser

from models.user_knn.model import UserKNN
from models.most_popular.model import MostPopular


config = configparser.ConfigParser()
config.read('config.ini')

user_knn = UserKNN.from_config(config=config)
most_popular = MostPopular.from_config(config=config)
