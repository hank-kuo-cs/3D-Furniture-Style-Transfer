import os
import re
from tqdm import tqdm
from ...object import Furniture, TripletFurniture
from ...config import config


class TripletFurnitureLoader:
    def __init__(self, dataset_type: str):
        assert dataset_type == 'train' or dataset_type == 'test'

        self.dataset_type = dataset_type
        self.object_dataset_path = config.dataset.shape_net_dataset_path
        self.crowdsource_data = self._load_crowdsource_data()
        self.triplet_furniture_list = self._load_triplet_furniture_data()

    def _load_triplet_furniture_data(self) -> list:
        triplet_furniture_list = []

        sample_data = self.get_sample_data()
        positive_data = self.get_positive_data()
        negative_data = self.get_negative_data()

        assert len(sample_data) == len(positive_data) == len(negative_data)

        for i in tqdm(range(len(sample_data))):
            sample_furniture = self.get_furniture_from_data(sample_data[i])
            positive_furniture = self.get_furniture_from_data(positive_data[i])
            negative_furniture = self.get_furniture_from_data(negative_data[i])

            triplet_furniture = TripletFurniture(sample_furniture, positive_furniture, negative_furniture)
            triplet_furniture_list.append(triplet_furniture)

        return triplet_furniture_list

    def _load_crowdsource_data(self):
        crowdsource_path = os.path.join(config.dataset.crowdsource_dataset_path, self.dataset_type) + '.txt'

        with open(crowdsource_path, 'r') as f:
            crowdsource_data = f.read()

        return crowdsource_data

    def get_sample_data(self) -> list:
        return re.findall(r's [a-z0-9]+ [0-9]', self.crowdsource_data)

    def get_positive_data(self) -> list:
        return re.findall(r'p [a-z0-9]+ [0-9]', self.crowdsource_data)

    def get_negative_data(self) -> list:
        return re.findall(r'n [a-z0-9]+ [0-9]', self.crowdsource_data)

    @staticmethod
    def get_furniture_from_data(data_line: str) -> Furniture:
        furniture_id, furniture_class = re.findall(r'[a-z] ([a-z0-9]+) ([0-9])', data_line)[0]
        furniture = Furniture(furniture_id=furniture_id, furniture_class=int(furniture_class))

        return furniture
