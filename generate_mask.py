import os
import cv2
from tqdm import tqdm
from glob import glob
from FurnitureStyleTransfer.renderer import Pytorch3DRenderer
from FurnitureStyleTransfer.object import Furniture, FurnitureClass
from FurnitureStyleTransfer.config import config


SHAPE_DATASET_PATH = config.dataset.shape_net_dataset_path


def generate_mask_dataset():
    class_names = FurnitureClass.class_names

    for class_name in class_names:
        print('\nGenerate mask images of "%s" dataset' % class_name)
        objects_of_one_class_dataset_path = glob(os.path.join(SHAPE_DATASET_PATH, class_name, '*'))

        for object_path in tqdm(objects_of_one_class_dataset_path):
            render_15_mask_images(class_name, object_path)


def render_15_mask_images(class_name: str, object_path: str):
    assert isinstance(class_name, str)
    assert isinstance(object_path, str)

    mask_images_path = os.path.join(object_path, 'mask_images')
    obj_path = os.path.join(object_path, 'model.obj')

    if os.path.exists(mask_images_path):
        return
    os.makedirs(mask_images_path, exist_ok=True)

    renderer = Pytorch3DRenderer(obj_path)

    for i in range(15):
        azim = i * (360.0 / 15)
        renderer.set_cameras(2, 15.0, azim, (0, 0, 0), (0, 1, 0), is_degree=True)
        mask_image = renderer.render_mask_image()
        mask_image = renderer.refine_mask(mask_image)

        img_path = '%s/mask-%d.png' % (mask_images_path, i)

        cv2.imwrite(img_path, mask_image)


if __name__ == '__main__':
    generate_mask_dataset()
