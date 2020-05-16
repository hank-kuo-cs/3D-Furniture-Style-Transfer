import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .loader import TripletFurnitureLoader
from ...config import config


class TripletFurnitureDataset(Dataset):
    def __init__(self, dataset_type):
        self.device = config.cuda.device
        self.dataset_type = dataset_type
        self.data_loader = TripletFurnitureLoader(dataset_type)

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
        sample_images = self.get_images(self.data_loader.triplet_furniture_list[item][0].images_path)
        positive_images = self.get_images(self.data_loader.triplet_furniture_list[item][1].images_path)
        negative_images = self.get_images(self.data_loader.triplet_furniture_list[item][2].images_path)

        return sample_images, positive_images, negative_images

    def get_images(self, images_path: list) -> list:
        images = []

        for img_path in images_path:
            img = Image.open(img_path)
            img = self.refine_img_data(img)
            images.append(img)

        return images

    def refine_img_data(self, img: np.ndarray) -> torch.Tensor:
        img = self.transform(img)
        img = img.to(self.device)

        return img
