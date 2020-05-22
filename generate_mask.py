import os
import cv2
from tqdm import tqdm
from FurnitureStyleTransfer.renderer import Pytorch3DRenderer
from FurnitureStyleTransfer.data import TripletFurnitureDataset
from FurnitureStyleTransfer.object import Furniture
from FurnitureStyleTransfer.config import config


IMAGE_DATASET_PATH = config.dataset.furniture_images_dataset_path


def generate_mask_dataset():
    train_loader = TripletFurnitureDataset('train').data_loader
    train_triplet_furniture_list = train_loader.triplet_furniture_list

    test_loader = TripletFurnitureDataset('test').data_loader
    test_triplet_furniture_list = test_loader.triplet_furniture_list

    print('Generate train mask dataset..')
    for triplet_furniture in tqdm(train_triplet_furniture_list):
        for i in range(3):
            render_15_mask_images(triplet_furniture[i])

    print('Generate test mask dataset..')
    for triplet_furniture in tqdm(test_triplet_furniture_list):
        for i in range(3):
            render_15_mask_images(triplet_furniture[i])


def render_15_mask_images(furniture: Furniture):
    assert isinstance(furniture, Furniture)

    mask_images_path = get_mask_images_path(furniture)

    if os.path.exists(mask_images_path):
        return
    os.makedirs(mask_images_path, exist_ok=True)

    renderer = Pytorch3DRenderer(furniture.obj_path)

    for i in range(15):
        azim = i * (360.0 / 15)
        renderer.set_cameras(2, 15.0, azim, (0, 0, 0), (0, 1, 0), is_degree=True)
        mask_image = renderer.render_mask_image()
        mask_image = renderer.refine_mask(mask_image)

        img_path = '%s/mask-%d.png' % (mask_images_path, i)

        cv2.imwrite(img_path, mask_image)


def get_mask_images_path(furniture) -> str:
    return os.path.join(IMAGE_DATASET_PATH, furniture.dataset_type, furniture.furniture_class_str, '3dw', furniture.furniture_id, 'mask_images')


if __name__ == '__main__':
    generate_mask_dataset()
