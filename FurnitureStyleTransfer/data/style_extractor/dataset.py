import os
import pickle
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .loader import TripletFurnitureLoader
from ...config import config


class TripletFurnitureDataset(Dataset):
    def __init__(self, dataset_type):
        self.device = config.cuda.device
        self.dataset_type = dataset_type
        self.data_loader = self._get_data_loader()

        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_loader.triplet_furniture_list)

    def __getitem__(self, item) -> (list, list, list):
        sample_images = self._get_images(self.data_loader.triplet_furniture_list[item][0].images_path)
        positive_images = self._get_images(self.data_loader.triplet_furniture_list[item][1].images_path)
        negative_images = self._get_images(self.data_loader.triplet_furniture_list[item][2].images_path)

        return sample_images, positive_images, negative_images

    def _get_images(self, images_path: list) -> list:
        images = []

        for img_path in images_path:
            img = Image.open(img_path)
            img = self.transform(img)
            images.append(img)

        return images

    def _get_data_loader(self):
        pickle_path = os.path.join(config.dataset.furniture_images_dataset_path,
                                   'triplet_images_%s_loader.pickle' % self.dataset_type)

        if os.path.exists(pickle_path):
            logging.info('Found saved data loader pickle file')
            with open(pickle_path, 'rb') as f:
                data_loader = pickle.load(f)
        else:
            logging.info('Cannot found saved data loader pickle file, load data...')
            data_loader = TripletFurnitureLoader(self.dataset_type)

            f = open(pickle_path, 'wb')
            pickle.dump(data_loader, f)
            f.close()

        return data_loader
